#!/bin/bash -l

DATASET="--train-dir /home/wang4538/DGMS-master/CIFAR10/train/ --val-dir /home/wang4538/DGMS-master/CIFAR10/val/ --num-classes 10"
MODEL="--network resnet18 --mask --empirical True"
WD=5e-4
TEMP=0.01
K=4
LR=5e-5
DATA_NAME="cifar10"
MODEL_NAME="resnet18"
EPOCHS="2ep"
FINAL_LR=2e-8
EVAL_INTERV='1ep'
SEED=10
FREEZE="--freeze_weight"
INIT_METHOD='quantile'


sbatch --time=4:00:00 --nodes=1 --gpus-per-node=1 --mem-per-gpu=40g <<EOT
#!/bin/bash -l

#SBATCH --output /home/wang4538/DGMS-master/out/%j.out
#SBATCH --error /home/wang4538/DGMS-master/out/%j.out

nvidia-smi
python ../main.py $DATASET $MODEL $RESUME $GPU $FREEZE --K ${K} --tau ${TEMP} --dataset ${DATA_NAME} --weight_decay ${WD} \
       --lr ${LR} --duration ${EPOCHS} --t_warmup "0.1dur" --alpha_f ${FINAL_LR} --seed ${SEED} --init_method ${INIT_METHOD} \
       --run_name debug --autoresume --eval_interval ${EVAL_INTERV} \
       --save_folder /scratch/gilbreth/wang4538/DGMS/Debug/${DATA_NAME}_${MODEL_NAME}/K${K}_temp${TEMP}_LR${LR}_F${FINAL_LR}_WD${WD}

EOT