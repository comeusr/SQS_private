#!/bin/bash -l

DATASET="--train-dir /home/wang4538/DGMS-master/CIFAR10/train/ --val-dir /home/wang4538/DGMS-master/CIFAR10/val/ --num-classes 10"
MODEL="--network resnet56 --mask"
WD=7e-5
TEMP=0.001
K=5
LR=5e-5
DATA_NAME="cifar10"
MODEL_NAME="resnet56"
EPOCHS="25ep"
FINAL_LR=0.006
EVAL_INTERV='1ep'
SEED=428
FREEZE="--freeze_weight"
INIT_METHOD='k-means'
INIT_SPARSITY=0.0
FINAL_SPARSITY=0.50
PRUNE_END='15ep'
PRUNE_TEMP=0.017
WARM_UP='1ep'
PRUNE_INIT_LR=0.012

sbatch --time=01:00:00 --nodes=1 --gpus-per-node=1 --mem-per-gpu=40g <<EOT
#!/bin/bash -l

#SBATCH --output /home/wang4538/DGMS-master/out/%j.out
#SBATCH --error /home/wang4538/DGMS-master/out/%j.out

nvidia-smi
python ../main.py $DATASET $MODEL $RESUME $GPU --K ${K} --tau ${TEMP} --dataset ${DATA_NAME} --weight_decay ${WD} --sample \
       --lr ${LR} --duration ${EPOCHS} --t_warmup WARM_UP --alpha_f ${FINAL_LR} --seed ${SEED} --init_method ${INIT_METHOD} \
       --run_name sample_${DATA_NAME}_${MODEL_NAME}_K${K}_KL_SPAS${FINAL_SPARSITY}_temp${TEMP}_prtemp${PRUNE_TEMP}_LR${LR}_prLR${PRUNE_INIT_LR}_F${FINAL_LR}_WD${WD} \
       --autoresume --eval_interval ${EVAL_INTERV} --prune_scale ${PRUNE_TEMP} --prune_init_lr ${PRUNE_INIT_LR} \
       --init_sparsity ${INIT_SPARSITY} --final_sparsity ${FINAL_SPARSITY} --prune_end ${PRUNE_END} --prune \
       --save_folder /scratch/gilbreth/wang4538/DGMS/Run/${INIT_METHOD}${DATA_NAME}_${MODEL_NAME}/K${K}_temp${TEMP}_LR${LR}_prLR${PRUNE_INIT_LR}_F${FINAL_LR}_WD${WD}_2

EOT