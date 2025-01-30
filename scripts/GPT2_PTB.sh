#!/bin/bash -l

DATASET="--dataset_name wikitext-103-v1"
WD=5e-7
TEMP=0.0005
K=16
LR=2e-5
EPOCHS=3
FINAL_LR=0.01
EVAL_INTERV='1ep'
SEED=428
INIT_METHOD='k-means'
INIT_SPARSITY=0.0
FINAL_SPARSITY=0.75
PRUNE_END=2.5
PRUNE_TEMP=0.01
WARM_UP=0.5
PRUNE_INIT_LR=0.01
SIGMA=3
MODEL_NAME='Qwen_0.5b'

# CIFAR100_K${K}_KL_SPAS${FINAL_SPARSITY}_temp${TEMP}_LR${LR}_F${FINAL_LR}_WD${WD}
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

module load anaconda
module load cuda/12.1.1
conda activate py311

sbatch --time=04:00:00 --nodes=1 --gpus-per-node=1 --mem-per-gpu=40g --job-name=${MODEL_NAME} --constraint="A100"<<EOT
#!/bin/bash -l

#SBATCH --output /home/wang4538/DGMS-master/out/%j.out
#SBATCH --error /home/wang4538/DGMS-master/err/%j.err

nvidia-smi
accelerate launch ../GPT2_exp.py $DATASET  --K ${K} --tau ${TEMP} --weight_decay ${WD} --prune --model_name ${MODEL_NAME} \
       --lr ${LR} --duration ${EPOCHS} --alpha_f ${FINAL_LR} --seed ${SEED} --init_method ${INIT_METHOD} \
       --run_name ${MODEL_NAME}_K${K}_KL_SPAS${FINAL_SPARSITY}_temp${TEMP}_LR${LR}_PRTEMP${PRUNE_TEMP}_WD${WD}_SIGMA${SIGMA}  \
       --autoresume --eval_interval ${EVAL_INTERV} --prune_scale ${PRUNE_TEMP} --prune_start ${WARM_UP} --sigma ${SIGMA} \
       --init_sparsity ${INIT_SPARSITY} --final_sparsity ${FINAL_SPARSITY} --prune_end ${PRUNE_END} \
       --save_folder /scratch/gilbreth/wang4538/DGMS/Run/${INIT_METHOD}${DATA_NAME}_${MODEL_NAME}/K${K}_temp${TEMP}_LR${LR}_F${FINAL_LR}_WD${WD}

EOT