#!/bin/bash -l

DATASET="--dataset_name squad"
WD=5e-7
TEMP=0.0005
K=4
LR=2e-5
EPOCHS=20
FINAL_LR=0.01
EVAL_INTERV='1ep'
SEED=428
INIT_METHOD='k-means'
INIT_SPARSITY=0.0
FINAL_SPARSITY=0.5
PRUNE_END='2ep'
PRUNE_TEMP=0.01
WARM_UP='1ep'
PRUNE_INIT_LR=0.01
SIGMA=3

# CIFAR100_K${K}_KL_SPAS${FINAL_SPARSITY}_temp${TEMP}_LR${LR}_F${FINAL_LR}_WD${WD}


sbatch --time=04:00:00 --nodes=1 --gpus-per-node=1 --mem-per-gpu=40g <<EOT
#!/bin/bash -l

#SBATCH --output /home/wang4538/DGMS-master/out/%j.out
#SBATCH --error /home/wang4538/DGMS-master/out/%j.out

nvidia-smi
python ../bert_exp.py $DATASET  --K ${K} --tau ${TEMP} --weight_decay ${WD} \
       --lr ${LR} --duration ${EPOCHS} --alpha_f ${FINAL_LR} --seed ${SEED} --init_method ${INIT_METHOD} \
       --run_name nonF_Squad_K${K}_KL_SPAS${FINAL_SPARSITY}_temp${TEMP}_LR${LR}_PRTEMP${PRUNE_TEMP}_WD${WD}_SIGMA${SIGMA}  \
       --autoresume --eval_interval ${EVAL_INTERV} --prune_scale ${PRUNE_TEMP} --warm_up ${WARM_UP} --sigma ${SIGMA} \
       --init_sparsity ${INIT_SPARSITY} --final_sparsity ${FINAL_SPARSITY} --prune_end ${PRUNE_END} \
       --save_folder /scratch/gilbreth/wang4538/DGMS/Run/${INIT_METHOD}${DATA_NAME}_BERT/K${K}_temp${TEMP}_LR${LR}_F${FINAL_LR}_WD${WD}

EOT