#!/bin/bash -l

DATASET="--train-dir /home/wang4538/DGMS-master/CIFAR10/train/ --val-dir /home/wang4538/DGMS-master/CIFAR10/val/ -d cifar10 --num-classes 10"
GENERAL="--lr 2e-5 --batch-size 128 --epochs 350 --workers 4 --base-size 32 --crop-size 32 --nesterov"
INFO="--checkname resnet562bit --lr-scheduler one-cycle"
MODEL="--network resnet56 --mask --K 4 --weight-decay 5e-4 --empirical True"
PARAMS="--tau 0.01"
RESUME="--show-info"
GPU="--gpu-ids 0"

sbatch --time=4:00:00 --nodes=1 --gpus-per-node=1 <<EOT
#!/bin/bash -l

#SBATCH --output /home/wang4538/DGMS-master/out/%j.out
#SBATCH --error /home/wang4538/DGMS-master/out/%j.out

nvidia-smi
python ../main.py $DATASET $GENERAL $MODEL $INFO $PARAMS $RESUME $GPU

EOT


