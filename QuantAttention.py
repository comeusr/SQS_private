from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import config as cfg
from modeling.DGMS.GMM import *

from typing import List, Optional, Tuple, Union

from transformers.models.bert.modeling_bert import load_tf_weights_in_bert, \
    BertSelfAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention


class CustomizeBertSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)

        self.dropout_prob = config.attention_probs_dropout_prob
        self.require_contiguous_qkv = False

        self.is_normal = cfg.IS_NORMAL

        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU
    
    def init_mask_params(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        self.query.sub_distribution = gmm_approximation(self.k_level, self.query.weight, self.temperature, init_method, sigma)
        self.key.sub_distribution = gmm_approximation(self.k_level, self.key.weight, self.temperature, init_method, sigma)
        self.value.sub_distribution = gmm_approximation(self.k_level, self.value.weight, self.temperature, init_method, sigma)

    def getMu(self):
        return (self.query.sub_distribution.mu.detach().data.cpu().numpy(), 
                self.key.sub_distribution.mu.detach().data.cpu().numpy(), 
                self.value.sub_distribution.mu.detach().data.cpu().numpy())
    
    def get_Sweight(self):
        # soft quantized weights during training
        with torch.no_grad():
            return (self.query.sub_distribution(weights=self.query.weight, train=True),
                    self.key.sub_distribution(weights=self.key.weight, train=True),
                    self.value.sub_distribution(weights=self.value.weight, train=True))

    def get_Pweight(self):
        # hard quantized weights during inference
        with torch.no_grad():
            return (self.query.sub_distribution(weights=self.query.weight, train=True),
                    self.key.sub_distribution(weights=self.key.weight, train=True),
                    self.value.sub_distribution(weights=self.value.weight, train=True))
        
    def QuantizedWeights(self):
        # Quantized weight from the given sub distribution.

        if cfg.IS_TRAIN:
            query_weight = self.query.sub_distribution(weights=self.query.weight, train=True)
            key_weight = self.key.sub_distribution(weights=self.key.weight, train=True)
            value_weight = self.value.sub_distribution(weights=self.value.weight, train=True)
        else:
            query_weight = self.query.sub_distribution(weights=self.query.weight, train=False)
            key_weight = self.key.sub_distribution(weights=self.key.weight, train=False)
            value_weight = self.value.sub_distribution(weights=self.value.weight, train=False)

        return query_weight, key_weight, value_weight

    def softforward(self, 
                query_weight: torch.Tensor,
                key_weight: torch.Tensor,
                value_weight: torch.Tensor,
                hidden_states: torch.Tensor, 
                attention_mask: torch.FloatTensor | None = None, 
                head_mask: torch.FloatTensor | None = None, 
                encoder_hidden_states: torch.FloatTensor | None = None, 
                encoder_attention_mask: torch.FloatTensor | None = None, 
                past_key_value: Tuple[Tuple[torch.FloatTensor]] | None = None, 
                output_attentions: bool | None = False) -> Tuple[torch.Tensor]:

        mixed_query_layer = F.linear(hidden_states, weight=query_weight)
        # print('Debug Key weight')
        # print(key_weight)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(F.linear(encoder_hidden_states, key_weight))
            value_layer = self.transpose_for_scores(F.linear(encoder_hidden_states, value_weight))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(F.linear(hidden_states, key_weight))
            value_layer = self.transpose_for_scores(F.linear(hidden_states, value_weight))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(F.linear(hidden_states, key_weight))
            value_layer = self.transpose_for_scores(F.linear(hidden_states, value_weight))  
        
        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: torch.FloatTensor | None = None, 
                head_mask: torch.FloatTensor | None = None, 
                encoder_hidden_states: torch.FloatTensor | None = None, 
                encoder_attention_mask: torch.FloatTensor | None = None, 
                past_key_value: Tuple[Tuple[torch.FloatTensor]] | None = None, 
                output_attentions: bool | None = False) -> Tuple[torch.Tensor]:
        if cfg.IS_NORMAL:
            return super().forward(hidden_states, 
                                   attention_mask, 
                                   head_mask, 
                                   encoder_hidden_states, 
                                   encoder_attention_mask, 
                                   past_key_value, 
                                   output_attentions)
        else:
            Qweight, Kweight, VWeight = self.QuantizedWeights()
            return self.softforward(Qweight,
                                Kweight,
                                VWeight,
                                hidden_states, 
                                attention_mask, 
                                head_mask, 
                                encoder_hidden_states, 
                                encoder_attention_mask, 
                                past_key_value, 
                                output_attentions)
    

class CustomizGPT2SdpaAttention(GPT2SdpaAttention):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

        self.is_normal = cfg.IS_NORMAL

        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU

    
    def init_mask_params(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        self.c_attn.sub_distribution = gmm_approximation(self.k_level, self.c_attn.weight, self.temperature, init_method, sigma)
        self.c_proj.sub_distribution = gmm_approximation(self.k_level, self.c_proj.weight, self.temperature, init_method, sigma)

    def get_Sweight(self):
        # soft quantized weights during training
        with torch.no_grad():
            return (self.c_attn.sub_distribution(weights=self.c_attn.weight, train=True),
                    self.c_proj.sub_distribution(weights=self.c_proj.weight, train=True))
    
    def QuantizedWeights(self):
        if cfg.IS_TRAIN:
            c_attn_weights = self.c_attn.sub_distribution(weights=self.c_attn.weight, train=True)
            c_proj_weights = self.c_proj.sub_distribution(weights=self.c_proj.weight, train=True)
        else:
            c_attn_weights = self.c_attn.sub_distribution(weights=self.c_attn.weight, train=True)
            c_proj_weights = self.c_proj.sub_distribution(weights=self.c_proj.weight, train=True)
        
        return c_attn_weights, c_proj_weights
        

    def softforward(
        self,
        c_attn_weights,
        c_proj_weights,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        bsz, q_len, _ = hidden_states.size()

        # Initial attention projections
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2SdpaAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = F.conv1d(hidden_states, c_attn_weights, self.c_attn.bias).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Optional kv caching
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache is True:
            present = (key, value)

        # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
        if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if attention_mask is None and q_len > 1 and not is_cross_attention else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.embed_dim)

        # Final projection
        attn_output = F.conv1d(attn_output, c_proj_weights, self.c_proj.bias)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present, None
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        if self.is_normal:
            return super().forward(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions
            )
        else:
            c_attn_weights, c_proj_weioghts = self.QuantizedWeights()
            return self.softforward(
                c_attn_weights=c_attn_weights,
                c_proj_weights=c_proj_weioghts,
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions
            )
    


        
        






        

# class BertLayer():
#     def __init__(self, config):
#         super().__init__()
#         self.chunk_size_feed_forward = config.chunk_size_feed_forward
#         self.seq_len_dim = 1
#         self.attention = BertAttention(config)
#         self.is_decoder = config.is_decoder
#         self.add_cross_attention = config.add_cross_attention
#         if self.add_cross_attention:
#             if not self.is_decoder:
#                 raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
#             self.crossattention = BertAttention(config, position_embedding_type="absolute")
#         self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)

#     def forward():
#         return 

# class BertEncoder(nn.Module):

#     def __init__(self, config, **kwargs) -> None:
#         super().__init__()
#         self.config = config
#         self.layer = nn.ModuleList([BertLayer(config) for _ in range(config['num_hidden_layers'])]) 
#         self.gradient_checkpointing = False

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.FloatTensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = False,
#         output_hidden_states: Optional[bool] = False,
#         return_dict: Optional[bool] = True,
#     ):
        
#         for i, layer_module in enumerate(self.layer):
#             if self.output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
#                                          encoder_attention_mask)
#             hidden_states = layer_outputs[0]

#             if self.output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)

#         # Add last layer
#         if self.output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         outputs = (hidden_states,)
#         if self.output_hidden_states:
#             outputs = outputs + (all_hidden_states,)
#         if self.output_attentions:
#             outputs = outputs + (all_attentions,)
#         return outputs  # last-layer hidden state, (all hidden states), (all attentions)



# class BertPreTrainedModel(PreTrainedModel):
#     config_class = BertConfig
#     # pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
#     load_tf_weights = load_tf_weights_in_bert
#     base_model_prefix = "bert"
#     supports_gradient_checkpointing = True
#     _supports_sdpa = True

#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, nn.Linear):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

# class BertModel(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertModel, self).__init__(config)
#         self.config = config

#         self.embeddings = BertEmbeddings(config)
#         self.encoder = BertEncoder(config)
#         self.pooler = BertPooler(config)

#         self.init_weights()

#     def get_input_embeddings(self):
#         return self.embeddings.word_embeddings

#     def set_input_embeddings(self, value):
#         self.embeddings.word_embeddings = value

#     def _prune_heads(self, heads_to_prune):
#         """ Prunes heads of the model.
#             heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
#             See base class PreTrainedModel
#         """
#         for layer, heads in heads_to_prune.items():
#             self.encoder.layer[layer].attention.prune_heads(heads)

#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
#                 head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         device = input_ids.device if input_ids is not None else inputs_embeds.device

#         if attention_mask is None:
#             attention_mask = torch.ones(input_shape, device=device)
#         if token_type_ids is None:
#             token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

#         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # ourselves in which case we just need to make it broadcastable to all heads.
#         if attention_mask.dim() == 3:
#             extended_attention_mask = attention_mask[:, None, :, :]
#         elif attention_mask.dim() == 2:
#             # Provided a padding mask of dimensions [batch_size, seq_length]
#             # - if the model is a decoder, apply a causal mask in addition to the padding mask
#             # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
#             if self.config.is_decoder:
#                 batch_size, seq_length = input_shape
#                 seq_ids = torch.arange(seq_length, device=device)
#                 causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
#                 causal_mask = causal_mask.to(
#                     torch.long)  # not converting to long will cause errors with pytorch version < 1.3
#                 extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
#             else:
#                 extended_attention_mask = attention_mask[:, None, None, :]
#         else:
#             raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape,
#                                                                                                         attention_mask.shape))

#         # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#         # masked positions, this operation will create a tensor which is 0.0 for
#         # positions we want to attend and -10000.0 for masked positions.
#         # Since we are adding it to the raw scores before the softmax, this is
#         # effectively the same as removing these entirely.
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         # If a 2D ou 3D attention mask is provided for the cross-attention
#         # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
#         if self.config.is_decoder and encoder_hidden_states is not None:
#             encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

#             if encoder_attention_mask.dim() == 3:
#                 encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
#             elif encoder_attention_mask.dim() == 2:
#                 encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
#             else:
#                 raise ValueError(
#                     "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
#                         encoder_hidden_shape,
#                         encoder_attention_mask.shape))

#             encoder_extended_attention_mask = encoder_extended_attention_mask.to(
#                 dtype=next(self.parameters()).dtype)  # fp16 compatibility
#             encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
#         else:
#             encoder_extended_attention_mask = None

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         if head_mask is not None:
#             if head_mask.dim() == 1:
#                 head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#                 head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
#             elif head_mask.dim() == 2:
#                 head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
#                     -1)  # We can specify head_mask for each layer
#             head_mask = head_mask.to(
#                 dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
#         else:
#             head_mask = [None] * self.config.num_hidden_layers

#         embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,
#                                            token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
#         encoder_outputs = self.encoder(embedding_output,
#                                        attention_mask=extended_attention_mask,
#                                        head_mask=head_mask,
#                                        encoder_hidden_states=encoder_hidden_states,
#                                        encoder_attention_mask=encoder_extended_attention_mask)
#         sequence_output = encoder_outputs[0]
#         pooled_output = self.pooler(sequence_output)

#         outputs = (sequence_output, pooled_output,) + encoder_outputs[
#                                                       1:]  # add hidden_states and attentions if they are here
#         return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
    

