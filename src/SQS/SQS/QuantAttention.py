from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import SQS.config as cfg
from SQS.modeling.DGMS.GMM import *

from typing import List, Optional, Tuple, Union

from transformers.models.bert.modeling_bert import load_tf_weights_in_bert, \
    BertSelfAttention, BertSelfOutput
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Attention
from transformers.models.opt.modeling_opt import OptFlashAttention2
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv, Qwen2RotaryEmbedding, eager_attention_forward
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS



logger = logging.get_logger(__name__)


class CustomizeBertSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)

        self.is_normal = cfg.IS_NORMAL

        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU
    
    def init_mask_params(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        self.dense.sub_distribution = gmm_approximation(self.k_level, self.dense.weight, self.temperature, init_method, sigma)

    def QuantizedWeights(self):
        # Quantized weight from the given sub distribution.

        if cfg.IS_TRAIN:
            _weight = self.dense.sub_distribution(weights=self.dense.weight, train=True)
        else:
            _weight = self.dense.sub_distribution(weights=self.dense.weight, train=False)
        
        return _weight

    def softforward(self, _weight:torch.Tensor, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = F.linear(hidden_states, _weight)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if cfg.IS_NORMAL:
            return super().forward(hidden_states, 
                                   input_tensor)
        else:
            _weight = self.QuantizedWeights()
            return self.softforward(_weight,
                                    hidden_states,
                                    input_tensor)
    
        

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
            return (self.query.sub_distribution(weights=self.query.weight, train=False),
                    self.key.sub_distribution(weights=self.key.weight, train=False),
                    self.value.sub_distribution(weights=self.value.weight, train=False))
        
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
    

class CustomizGPT2Attention(GPT2Attention):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

        self.is_normal = cfg.IS_NORMAL

        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU

    
    def init_mask_params(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        self.c_attn.sub_distribution = gmm_approximation(self.k_level, self.c_attn.weight, self.temperature, init_method, sigma)
        self.c_proj.sub_distribution = gmm_approximation(self.k_level, self.c_proj.weight, self.temperature, init_method, sigma)

        # self.c_attn = nn.Conv1d(in_channels, out_channels)


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

        # print("Original c_attn weight shape {}".format(self.c_attn.weight.shape))
        # print("Customize c_attn weight shape {}".format(c_attn_weights.shape))

        # Initial attention projections
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = F.linear(hidden_states, c_attn_weights.transpose(0,1), self.c_attn.bias).split(self.split_size, dim=2)

        # self.c_attn.weight
        # self.c_attn(hidden_states)

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
        # attn_output = self.c_proj(attn_output)
        attn_output = F.linear(attn_output, c_proj_weights.transpose(0,1), self.c_proj.bias)
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
    

class CustomizedOPTFlashAttention2(OptFlashAttention2):
    def __init__(self, config, is_decoder=False):
        super().__init__(config=config, is_decoder=is_decoder)

        self.is_normal = cfg.IS_NORMAL
        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU
    
    def init_mask_params(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        self.k_proj.sub_distribution = gmm_approximation(self.k_level, self.k_proj.weight, self.temperature, init_method, sigma)    
        self.v_proj.sub_distribution = gmm_approximation(self.k_level, self.v_proj.weight, self.temperature, init_method, sigma)    
        self.q_proj.sub_distribution = gmm_approximation(self.k_level, self.q_proj.weight, self.temperature, init_method, sigma)    
        self.out_proj.sub_distribution = gmm_approximation(self.k_level, self.out_proj.weight, self.temperature, init_method, sigma)    
    
    def get_Sweight(self):
        with torch.no_grad():
            return (self.k_proj.sub_distribution(weights=self.k_proj.weight, train=True),
                    self.v_proj.sub_distribution(weights=self.v_proj.weight, train=True),
                    self.q_proj.sub_distribution(weights=self.q_proj.weight, train=True),
                    self.out_proj.sub_distribution(weights=self.out_proj.weight, train=True))
    
    def QuantizedWeights(self):
        if cfg.IS_TRAIN:
            k_weights = self.k_proj.sub_distribution(weights=self.k_proj.weight, train=True)
            v_weights = self.v_proj.sub_distribution(weights=self.v_proj.weight, train=True)
            q_weights = self.q_proj.sub_distribution(weights=self.q_proj.weight, train=True)
            out_weights = self.out_proj.sub_distribution(weights=self.out_proj.weight, train=True)
        else:
            k_weights = self.k_proj.sub_distribution(weights=self.k_proj.weight, train=False)
            v_weights = self.v_proj.sub_distribution(weights=self.v_proj.weight, train=False)
            q_weights = self.q_proj.sub_distribution(weights=self.q_proj.weight, train=False)
            out_weights = self.out_proj.sub_distribution(weights=self.out_proj.weight, train=False)

        return k_weights, v_weights, q_weights, out_weights
    
    def softforward(self, 
                    k_weights, 
                    v_weights, 
                    q_weights, 
                    out_weights, 
                    hidden_states: torch.Tensor,
                    key_value_states: Optional[torch.Tensor] = None,
                    past_key_value: Optional[Tuple[torch.Tensor]] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    layer_head_mask: Optional[torch.Tensor] = None,
                    output_attentions: bool = False,
                    position_ids: Optional[torch.Tensor] = None,
                    ):
        is_cross_attention = key_value_states is not None

        bsz, _, _ = hidden_states.size()

        # get query proj
        query_states = F.linear(hidden_states, q_weights, self.q_proj.bias)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(F.linear(key_value_states, k_weights, self.k_proj.bias), -1, bsz)
            value_states = self._shape(F.linear(key_value_states, v_weights, self.v_proj.bias), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(F.linear(hidden_states, k_weights, self.k_proj.bias), -1, bsz)
            value_states = self._shape(F.linear(hidden_states, v_weights, self.v_proj.bias), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(F.linear(hidden_states, k_weights, self.k_proj.bias), -1, bsz)
            value_states = self._shape(F.linear(hidden_states, v_weights, self.v_proj.bias), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_length = query_states.shape[1]
        tgt_len = key_states.shape[-2]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query_states = query_states.view(bsz, query_length, self.num_heads, self.head_dim)
        key_states = key_states.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
        value_states = value_states.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

        attn_dropout = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            position_ids=position_ids,
            dropout=attn_dropout,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
        # attn_output = self.out_proj(attn_weights_reshaped)
        attn_output = F.linear(attn_weights_reshaped, out_weights, self.out_proj.bias)

        if not output_attentions:
            attn_weights_reshaped = None

        return attn_output, attn_weights_reshaped, past_key_value


    def forward(self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if self.is_normal:
            return super().forward(hidden_states, key_value_states, past_key_value, attention_mask, layer_head_mask, output_attentions, position_ids)
        else:
            k_weights, v_weights, q_weights, out_weights = self.QuantizedWeights()
            return self.softforward(k_weights, v_weights, q_weights, out_weights, hidden_states, key_value_states, past_key_value, attention_mask, layer_head_mask, output_attentions, position_ids)


class CustomizedQwen2Attention(Qwen2Attention):

    def __init__(self, config, layer_idx=False):
        super().__init__(config=config, layer_idx=layer_idx)

        self.is_normal = cfg.IS_NORMAL
        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU

        # self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
    
    def init_mask_params(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        if torch.isnan(self.k_proj.weight).any():
            print("Original k_proj weight is nan")
        if torch.isnan(self.v_proj.weight).any():
            print("Original v_proj weight is nan")
        if torch.isnan(self.q_proj.weight).any():
            print("Original q_proj weight is nan")
        if torch.isnan(self.o_proj.weight).any():
            print("Original o_proj weight is nan")
        self.k_proj.sub_distribution = gmm_approximation(self.k_level, self.k_proj.weight, self.temperature, init_method, sigma)    
        self.v_proj.sub_distribution = gmm_approximation(self.k_level, self.v_proj.weight, self.temperature, init_method, sigma)    
        self.q_proj.sub_distribution = gmm_approximation(self.k_level, self.q_proj.weight, self.temperature, init_method, sigma)    
        self.o_proj.sub_distribution = gmm_approximation(self.k_level, self.o_proj.weight, self.temperature, init_method, sigma)    
    
    def get_Sweight(self):
        with torch.no_grad():
            return (self.k_proj.sub_distribution(weights=self.k_proj.weight, train=True),
                    self.v_proj.sub_distribution(weights=self.v_proj.weight, train=True),
                    self.q_proj.sub_distribution(weights=self.q_proj.weight, train=True),
                    self.o_proj.sub_distribution(weights=self.o_proj.weight, train=True))
    
    def QuantizedWeights(self):
        if cfg.IS_TRAIN:

            # print("-"*50+"In Training Fetch Quantized Weights"+"-"*50)
            if torch.isnan(self.k_proj.weight).any():
                print("Original k_proj weight is nan")
            if torch.isnan(self.v_proj.weight).any():
                print("Original v_proj weight is nan")
            if torch.isnan(self.q_proj.weight).any():
                print("Original q_proj weight is nan")
            if torch.isnan(self.o_proj.weight).any():
                print("Original o_proj weight is nan")
            
    
            k_weights = self.k_proj.sub_distribution(weights=self.k_proj.weight, train=True)
            v_weights = self.v_proj.sub_distribution(weights=self.v_proj.weight, train=True)
            q_weights = self.q_proj.sub_distribution(weights=self.q_proj.weight, train=True)
            o_weights = self.o_proj.sub_distribution(weights=self.o_proj.weight, train=True)
        else:
            if torch.isnan(self.k_proj.weight).any():
                print("Original k_proj weight is nan")
            if torch.isnan(self.v_proj.weight).any():
                print("Originalv_proj weight is nan")
            if torch.isnan(self.q_proj.weight).any():
                print("Original q_proj weight is nan")
            if torch.isnan(self.o_proj.weight).any():
                print("Original o_proj weight is nan")
            k_weights = self.k_proj.sub_distribution(weights=self.k_proj.weight, train=False)
            v_weights = self.v_proj.sub_distribution(weights=self.v_proj.weight, train=False)
            q_weights = self.q_proj.sub_distribution(weights=self.q_proj.weight, train=False)
            o_weights = self.o_proj.sub_distribution(weights=self.o_proj.weight, train=False)

        return k_weights, v_weights, q_weights, o_weights

    def softforward(
        self,
        k_weights, 
        v_weights, 
        q_weights, 
        o_weights, 
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)        

        # try:
        query_states = F.linear(hidden_states, q_weights.to(hidden_states.dtype), self.q_proj.bias).view(hidden_shape).transpose(1, 2)
            # print(hidden_states.dtype)
            # print(q_weights.dtype)
            # print(self.q_proj.bias.dtype)

        key_states = F.linear(hidden_states, k_weights.to(hidden_states.dtype), self.k_proj.bias).view(hidden_shape).transpose(1, 2)
        value_states = F.linear(hidden_states, v_weights.to(hidden_states.dtype), self.v_proj.bias).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = F.linear(attn_output, o_weights.to(attn_output.dtype))
        return attn_output, attn_weights
        

    
    def forward( self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if self.is_normal:
            return super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value,
                cache_position,
                **kwargs
            )
        else:
            k_weights, v_weights, q_weights, o_weights = self.QuantizedWeights()
            temp = self.softforward(
                k_weights, 
                v_weights, 
                q_weights, 
                o_weights, 
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value,
                cache_position,
                **kwargs
            )

            # print('-'*50+"Temp requires_grad: ", temp[0].requires_grad, "-"*50)
            if temp[0].isnan().any():
                print("Temp is nan")
            elif temp[0].isinf().any():
                print("Temp is inf")

            # temp[0].requires_grad = True
            # print('-'*50+"Temp requires_grad: ", temp[0].requires_grad, "-"*50)

            return temp


class CustomizedLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)

        self.is_normal = cfg.IS_NORMAL
        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU

    def init_mask_params(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        self.q_proj.sub_distribution = gmm_approximation(self.k_level, self.q_proj.weight, self.temperature, 32, init_method, sigma)
        self.k_proj.sub_distribution = gmm_approximation(self.k_level, self.k_proj.weight, self.temperature, 32, init_method, sigma)
        self.v_proj.sub_distribution = gmm_approximation(self.k_level, self.v_proj.weight, self.temperature, 32, init_method, sigma)
        self.o_proj.sub_distribution = gmm_approximation(self.k_level, self.o_proj.weight, self.temperature, 32, init_method, sigma)

    def QuantizedWeights(self, train):
        return self.q_proj.sub_distribution(weights=self.q_proj.weight, train=train), self.k_proj.sub_distribution(weights=self.k_proj.weight, train=train), self.v_proj.sub_distribution(weights=self.v_proj.weight, train=train), self.o_proj.sub_distribution(weights=self.o_proj.weight, train=train)

    def softforward( 
        self,
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = F.linear(hidden_states, q_weights.to(hidden_states.dtype), self.q_proj.bias).view(hidden_shape).transpose(1, 2)
        key_states = F.linear(hidden_states, k_weights.to(hidden_states.dtype), self.k_proj.bias).view(hidden_shape).transpose(1, 2)
        value_states = F.linear(hidden_states, v_weights.to(hidden_states.dtype), self.v_proj.bias).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = F.linear(attn_output, o_weights.to(attn_output.dtype))
        return attn_output, attn_weights

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
        ):
        if self.is_normal:
            return super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value,
                cache_position,
                **kwargs
            )
        else:
            q_weights, k_weights, v_weights, o_weights = self.QuantizedWeights(cfg.IS_TRAIN)
            return self.softforward(q_weights, 
                                    k_weights, 
                                    v_weights, 
                                    o_weights, 
                                    hidden_states, 
                                    position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)



class CustomizedLLamaMLP(LlamaMLP):
    def __init__(self, config, blocks=4, sigma=None):
        super().__init__(config)

        self.is_normal = cfg.IS_NORMAL
        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU
        self.blocks = blocks
        self.up_proj_size = self.up_proj.weight.size()
        self.down_proj_size = self.down_proj.weight.size()
        self.sorted_up_indices = None
        self.sorted_down_indices = None
        self.up_weight_num = self.up_proj.weight.numel()
        self.down_weight_num = self.down_proj.weight.numel()
        self.up_first_n=64
        self.up_last_n=64
        self.down_first_n=64
        self.down_last_n=64
    
    def reconstruct_weight(self, weights: List[torch.Tensor], inverse_sorted_indices, type='up'):
        # TODO: Restore the up_proj_weight given up_weight and self.sorted_up_indices
        # TODO: Restore the down_proj_weight given down_weight and self.sorted_down_indices

        # print("-"*50+"inverse_sorted_indices: ", inverse_sorted_indices.device, "-"*50)

        if type == 'up':
            return torch.cat(weights, dim=0)[inverse_sorted_indices].view(self.up_proj_size)
        else:
            return torch.cat(weights, dim=0)[inverse_sorted_indices].view(self.down_proj_size)

            
    
    def get_outlier_indices(self, scale=5):
        
        # Before Call this function, self.up_proj.weight and self.down_proj.weight need to be flattened and sorted
        
        fisrt_quantile_index = (self.up_weight_num-1)//4
        third_quantile_index = (self.up_weight_num-1)*3//4

        up_iqr = self.up_proj.weight.data[third_quantile_index]-self.up_proj.weight.data[fisrt_quantile_index]
        down_iqr = self.down_proj.weight.data[third_quantile_index]-self.down_proj.weight.data[fisrt_quantile_index]

        up_low_threshold = self.up_proj.weight.data[fisrt_quantile_index]-scale*up_iqr
        up_high_threshold = self.up_proj.weight.data[third_quantile_index]+scale*up_iqr

        down_low_threshold = self.down_proj.weight.data[fisrt_quantile_index]-scale*down_iqr
        down_high_threshold = self.down_proj.weight.data[third_quantile_index]+scale*down_iqr


        self.up_first_n = torch.searchsorted(self.up_proj.weight.data, up_low_threshold, right=True).item()
        self.up_last_n = self.up_weight_num - torch.searchsorted(self.up_proj.weight.data, up_high_threshold, right=True).item()

        self.down_first_n = torch.searchsorted(self.down_proj.weight.data, down_low_threshold, right=True).item()
        self.down_last_n = self.down_weight_num - torch.searchsorted(self.down_proj.weight.data, down_high_threshold, right=True).item()

        return
    
    def DGMS_INIT(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'

        up_flat_weight = self.up_proj.weight.flatten().data
        down_flat_weight = self.down_proj.weight.flatten().data

        self.up_proj.weight.data, sorted_up_indices = torch.sort(up_flat_weight)
        self.down_proj.weight.data, sorted_down_indices = torch.sort(down_flat_weight)

        self.up_argsort_indices = torch.argsort(sorted_up_indices).cpu()
        self.down_argsort_indices = torch.argsort(sorted_down_indices).cpu()

        del up_flat_weight, sorted_up_indices
        del down_flat_weight, sorted_down_indices

        torch.cuda.empty_cache()

        self.up_proj.sub_distribution_list = []
        self.down_proj.sub_distribution_list = []

        self.up_step_size = self.up_proj.weight.numel()//self.blocks
        self.down_step_size = self.down_proj.weight.numel()//self.blocks

        for block_idx in range(self.blocks):
            start = block_idx*self.up_step_size
            end = start+self.up_step_size

            self.up_proj.sub_distribution_list.append(gmm_approximation(self.k_level, self.up_proj.weight[start:end].contiguous(), self.temperature, 32, init_method, sigma).to(device=self.up_proj.weight.device))
            self.down_proj.sub_distribution_list.append(gmm_approximation(self.k_level, self.down_proj.weight[start:end].contiguous(), self.temperature, 32, init_method, sigma).to(device=self.down_proj.weight.device))

        self.up_proj.sub_distribution_list = nn.ModuleList(self.up_proj.sub_distribution_list)
        self.down_proj.sub_distribution_list = nn.ModuleList(self.down_proj.sub_distribution_list)

        
    def SQS_INIT(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'

        up_flat_weight = self.up_proj.weight.flatten().data
        down_flat_weight = self.down_proj.weight.flatten().data

        self.up_proj.weight.data, sorted_up_indices = torch.sort(up_flat_weight)
        self.down_proj.weight.data, sorted_down_indices = torch.sort(down_flat_weight)

        self.get_outlier_indices()

        self.up_argsort_indices = torch.argsort(sorted_up_indices).cpu()
        self.down_argsort_indices = torch.argsort(sorted_down_indices).cpu()

        del up_flat_weight, sorted_up_indices
        del down_flat_weight, sorted_down_indices

        torch.cuda.empty_cache()

        self.up_proj.sub_distribution_list = []
        self.down_proj.sub_distribution_list = []

        self.up_step_size = (self.up_proj.weight.numel()-self.up_first_n-self.up_last_n)//(self.blocks-2)
        self.down_step_size = (self.down_proj.weight.numel()-self.down_first_n-self.down_last_n)//(self.blocks-2)

        for block_idx in range(self.blocks):

            # print("-"*50+"up_mask shape: ", up_mask.shape, "-"*50)
            # print("-"*50+"up selectedweight shape {} ".format(self.up_proj.weight[up_mask].shape), "-"*50)
            if block_idx == 0:
                self.up_proj.sub_distribution_list.append(gmm_approximation(self.k_level, self.up_proj.weight[0:self.up_first_n].contiguous(), self.temperature, 20, init_method, sigma).to(device=self.up_proj.weight.device))
                self.down_proj.sub_distribution_list.append(gmm_approximation(self.k_level, self.down_proj.weight[0:self.down_first_n].contiguous(), self.temperature, 20, init_method, sigma).to(device=self.down_proj.weight.device))
            elif block_idx == self.blocks-1:
                self.up_proj.sub_distribution_list.append(gmm_approximation(self.k_level, self.up_proj.weight[self.up_weight_num-self.up_last_n:self.up_weight_num].contiguous(), self.temperature, 20, init_method, sigma).to(device=self.up_proj.weight.device))
                self.down_proj.sub_distribution_list.append(gmm_approximation(self.k_level, self.down_proj.weight[self.down_weight_num-self.down_last_n:self.down_weight_num].contiguous(), self.temperature, 20, init_method, sigma).to(device=self.down_proj.weight.device))
            else:
                if block_idx == self.blocks-2:
                    up_start = self.up_first_n+(block_idx-1)*self.up_step_size
                    up_end =  self.up_weight_num-self.up_last_n
                    down_start = self.down_first_n+(block_idx-1)*self.down_step_size
                    down_end = self.down_weight_num-self.down_last_n
                else:
                    up_start = self.up_first_n+(block_idx-1)*self.up_step_size
                    up_end = up_start+self.up_step_size
                    down_start = self.down_first_n+(block_idx-1)*self.down_step_size
                    down_end = down_start+self.down_step_size

                
                self.up_proj.sub_distribution_list.append(gmm_approximation(self.k_level, self.up_proj.weight[up_start:up_end].contiguous(), self.temperature, 20, init_method, sigma).to(device=self.up_proj.weight.device))
                self.down_proj.sub_distribution_list.append(gmm_approximation(self.k_level, self.down_proj.weight[down_start:down_end].contiguous(), self.temperature, 20, init_method, sigma).to(device=self.down_proj.weight.device))

        self.up_proj.sub_distribution_list = nn.ModuleList(self.up_proj.sub_distribution_list)
        self.down_proj.sub_distribution_list = nn.ModuleList(self.down_proj.sub_distribution_list)


    def DGMS_QuantizedWeights(self, train=True):
        up_weights = []
        down_weights = []

        for idx in range(self.blocks):
            start = idx*self.up_step_size
            end = start+self.up_step_size

            up_weights.append(self.up_proj.sub_distribution_list[idx](self.up_proj.weight[start:end].contiguous(), train=train))
            down_weights.append(self.down_proj.sub_distribution_list[idx](self.down_proj.weight[start:end].contiguous(), train=train))

        up_weights = self.reconstruct_weight(up_weights, self.up_argsort_indices, type='up')
        down_weights = self.reconstruct_weight(down_weights, self.down_argsort_indices, type='down')

        return up_weights, down_weights
            

    def SQS_QuantizedWeights(self, train=True):
        # Adaptive quantization
        up_weights = []
        down_weights = []

        for idx in range(self.blocks):

            if idx == 0:
                identity=False
                up_start = 0
                up_end = self.up_first_n
                down_start = 0
                down_end = self.down_first_n
                
            elif idx == self.blocks-1:
                identity=False
                up_start = self.up_weight_num-self.up_last_n
                up_end = self.up_weight_num
                down_start = self.down_weight_num-self.down_last_n
                down_end = self.down_weight_num

                
            elif idx == self.blocks-2:
                identity=False
                up_start = self.up_first_n+(idx-1)*self.up_step_size
                up_end =  self.up_weight_num-self.up_last_n
                down_start = self.down_first_n+(idx-1)*self.down_step_size
                down_end = self.down_weight_num-self.down_last_n

            else:
                identity=False
                up_start = self.up_first_n+(idx-1)*self.up_step_size
                up_end = up_start+self.up_step_size
                down_start = self.down_first_n+(idx-1)*self.down_step_size
                down_end = down_start+self.down_step_size
            
            if identity:
                up_weights.append(self.up_proj.sub_distribution_list[idx](self.up_proj.weight[up_start:up_end].contiguous()))
                down_weights.append(self.down_proj.sub_distribution_list[idx](self.down_proj.weight[down_start:down_end].contiguous()))
            else:
                up_weights.append(self.up_proj.sub_distribution_list[idx](weights=self.up_proj.weight[up_start:up_end].contiguous(), train=train))
                down_weights.append(self.down_proj.sub_distribution_list[idx](weights=self.down_proj.weight[down_start:down_end].contiguous(), train=train))

        up_weights = self.reconstruct_weight(up_weights, self.up_argsort_indices, type='up')
        down_weights = self.reconstruct_weight(down_weights, self.down_argsort_indices, type='down')
        # print("-"*50+"Quantization Time taken to reconstruct weights {:.4f}".format(time_end-time_start)+"-"*50)

        return up_weights, down_weights
    
    def softforward(self,
        up_weights,
        down_weights,
        x
    ):
        x = self.act_fn(self.gate_proj(x))*F.linear(x, up_weights.to(x.dtype), self.up_proj.bias)
        down_proj = F.linear(x, down_weights.to(x.dtype), self.down_proj.bias)
        return down_proj

    def forward(self, x):
        if self.is_normal:
            return super().forward(x)
        else:
            # print("-"*50+"CustomizedMLP Layer Forward"+"-"*50)
            if cfg.METHOD == "SQS":
                up_weights, down_weights = self.SQS_QuantizedWeights(cfg.IS_TRAIN)
            else:
                up_weights, down_weights = self.DGMS_QuantizedWeights(cfg.IS_TRAIN)
            return self.softforward(up_weights, down_weights, x)

