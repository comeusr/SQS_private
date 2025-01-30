#!/bin/bash -l

DATASET="--train-dir /scratch/gilbreth/wang4538/DGMS-master/CIFAR100/train/ --val-dir /scratch/gilbreth/wang4538/DGMS-master/CIFAR100/val/ --num-classes 100"
MODEL="--network resnet50 --mask"
WD=5e-7
TEMP=0.001
K=16
LR=5e-5
DATA_NAME="cifar100"
MODEL_NAME="resnet50"
EPOCHS="11ep"
FINAL_LR=0.007
EVAL_INTERV='1ep'
SEED=428
FREEZE="--freeze_weight"
INIT_METHOD='k-means'
INIT_SPARSITY=0.00
FINAL_SPARSITY=0.70
PRUNE_END='9ep'
PRUNE_TEMP=0.01
PRUNE_INIT_LR=0.012
WARM_UP='5ep'
AVERAGE_NUM=10
PRIOR="spike_slab"

sh <<EOT
#!/bin/bash -l

#SBATCH --output /home/wang4538/DGMS-master/out/%j.out
#SBATCH --error /home/wang4538/DGMS-master/out/%j.out

nvidia-smi
python ../main.py $DATASET $MODEL $RESUME $GPU --K ${K} --tau ${TEMP} --dataset ${DATA_NAME} --weight_decay ${WD} \
       --lr ${LR} --duration ${EPOCHS} --t_warmup "5ep" --alpha_f ${FINAL_LR} --seed ${SEED} --init_method ${INIT_METHOD} \
       --run_name DGMS_${DATA_NAME}_${MODEL_NAME}_K${K}_KL_SPAS${FINAL_SPARSITY}_temp${TEMP}_prtemp${PRUNE_TEMP}_LR${LR}_prLR${PRUNE_INIT_LR}_F${FINAL_LR}_WD${WD}_${AVERAGE_NUM} \
       --sample --average_num ${AVERAGE_NUM} --prune --prior ${PRIOR} \
       --autoresume --eval_interval ${EVAL_INTERV} --prune_scale ${PRUNE_TEMP} --warm_up ${WARM_UP} \
       --init_sparsity ${INIT_SPARSITY} --final_sparsity ${FINAL_SPARSITY} --prune_end ${PRUNE_END}  --prune_init_lr ${PRUNE_INIT_LR} \
       --save_folder /scratch/gilbreth/wang4538/DGMS/Run/${INIT_METHOD}${DATA_NAME}_${MODEL_NAME}/K${K}_temp${TEMP}_LR${LR}_F${FINAL_LR}_WD${WD}

EOT
