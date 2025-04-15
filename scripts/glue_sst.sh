DATASET="--dataset_name ptb_text_only"
TASK_NAME='sst2'
WD=5e-7
TEMP=0.001
K=16
LR=5e-8
EPOCHS=1
FINAL_LR=0.01
EVAL_INTERV='1ep'
SEED=428
INIT_METHOD='k-means'
INIT_SPARSITY=0.0
FINAL_SPARSITY=0.75
PRUNE_END=0.9
PRUNE_TEMP=0.01
WARM_UP=0.00
PRUNE_INIT_LR=0.01
SIGMA=3
MODEL_NAME='meta-llama/Llama-3.2-1B'
OPTIMIZER='sgd'
BATCH_SIZE=64
EVAL_BATCH_SIZE=256
MAX_LENGTH=384

# CIFAR100_K${K}_KL_SPAS${FINAL_SPARSITY}_temp${TEMP}_LR${LR}_F${FINAL_LR}_WD${WD}
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

module load cuda/12.1

conda activate LLM


sh <<EOT
#!/bin/bash -l

#SBATCH --job-name ${MODEL_NAME}normal
#SBATCH --output /home/wang4538/DGMS-master/out/%x_%j.out
#SBATCH --error /home/wang4538/DGMS-master/err/%x_%j.err


nvidia-smi
python ../glue_training.py $DATASET --task_name ${TASK_NAME} --K ${K} --tau ${TEMP} --weight_decay ${WD} --prune --model_name ${MODEL_NAME} \
       --lr ${LR} --duration ${EPOCHS} --alpha_f ${FINAL_LR} --seed ${SEED} --init_method ${INIT_METHOD} \
       --run_name ${MODEL_NAME}_K${K}_KL_SPAS${FINAL_SPARSITY}_temp${TEMP}_LR${LR}_PRTEMP${PRUNE_TEMP}_WD${WD}_SIGMA${SIGMA}  \
       --autoresume --eval_interval ${EVAL_INTERV} --prune_scale ${PRUNE_TEMP} --prune_start ${WARM_UP} --sigma ${SIGMA} \
       --init_sparsity ${INIT_SPARSITY} --final_sparsity ${FINAL_SPARSITY} --prune_end ${PRUNE_END} --optimizer ${OPTIMIZER} \
       --batch_size ${BATCH_SIZE} --max_seq_length ${MAX_LENGTH} --preprocessing_num_workers 4 \
       --eval_batch_size ${EVAL_BATCH_SIZE} --save_folder /scratch/gilbreth/wang4538/DGMS/Run/GLUE/compressed_${MODEL_NAME}/ 
EOT