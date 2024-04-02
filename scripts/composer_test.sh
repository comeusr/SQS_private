#!/bin/bash -l

DATASET="--train-dir /home/wang4538/DGMS-master/CIFAR10/train/ --val-dir /home/wang4538/DGMS-master/CIFAR10/val/ --num-classes 10"
MODEL="--network resnet18 --mask --weight-decay 5e-4 --empirical True"
PARAMS="--tau 0.01"
K=4
DATA_NAME="cifar10"
MODEL_NAME="resnet18"



sbatch --time=4:00:00 --nodes=1 --gpus-per-node=1 --mem-per-gpu=40g <<EOT
#!/bin/bash -l

#SBATCH --output /home/wang4538/DGMS-master/out/%j.out
#SBATCH --error /home/wang4538/DGMS-master/out/%j.out

nvidia-smi
python ../main.py $DATASET $MODEL $PARAMS $RESUME $GPU K=${K} \
       load_path=/scratch/gilbreth/wang4538/DGMS/${DATA_NAME}_${MODEL_NAME}/K${K}_temp

EOT