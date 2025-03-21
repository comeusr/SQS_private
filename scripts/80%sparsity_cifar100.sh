#!/bin/bash -l

DATASET="--train-dir /home/wang4538/DGMS-master/CIFAR100/train/ --val-dir /home/wang4538/DGMS-master/CIFAR100/val/ --num-classes 100"
MODEL="--network resnet18 --mask"
WD=5e-5
TEMP=0.0005
K=4
LR=1e-5
DATA_NAME="cifar100"
MODEL_NAME="resnet18"
EPOCHS="20ep"
FINAL_LR=0.01
EVAL_INTERV='1ep'
SEED=428
FREEZE="--freeze_weight"
INIT_METHOD='k-means'
INIT_SPARSITY=0.0
FINAL_SPARSITY=0.8
PRUNE_END='10ep'
PRUNE_TEMP=0.01
WARM_UP='1ep'
PRUNE_INIT_LR=0.01
SIGMA=5

# CIFAR100_K${K}_KL_SPAS${FINAL_SPARSITY}_temp${TEMP}_LR${LR}_F${FINAL_LR}_WD${WD}


sbatch --time=04:00:00 --nodes=1 --gpus-per-node=1 --mem-per-gpu=40g <<EOT
#!/bin/bash -l

#SBATCH --output /home/wang4538/DGMS-master/out/%j.out
#SBATCH --error /home/wang4538/DGMS-master/out/%j.out

nvidia-smi
python ../main.py $DATASET $MODEL $RESUME $GPU --K ${K} --tau ${TEMP} --dataset ${DATA_NAME} --weight_decay ${WD} --sample \
       --lr ${LR} --duration ${EPOCHS} --t_warmup "5ep" --alpha_f ${FINAL_LR} --seed ${SEED} --init_method ${INIT_METHOD} \
       --run_name nonF_sample_CIFAR100_K${K}_KL_SPAS${FINAL_SPARSITY}_temp${TEMP}_LR${LR}_PRTEMP${PRUNE_TEMP}_WD${WD}_SIGMA${SIGMA} --prune_init_lr ${PRUNE_INIT_LR} \
       --autoresume --eval_interval ${EVAL_INTERV} --prune_scale ${PRUNE_TEMP} --warm_up ${WARM_UP} --prior_sigma ${SIGMA} \
       --init_sparsity ${INIT_SPARSITY} --final_sparsity ${FINAL_SPARSITY} --prune_end ${PRUNE_END} --prune \
       --save_folder /scratch/gilbreth/wang4538/DGMS/Run/SAMPLE_${INIT_METHOD}${DATA_NAME}_${MODEL_NAME}/K${K}_temp${TEMP}_LR${LR}_F${FINAL_LR}_WD${WD}

EOT