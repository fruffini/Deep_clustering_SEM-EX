#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 2-4:00:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it



# Load modules
module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0

# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/envs/venv
source bin/activate



# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/src/models/DCEC

rep='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports'
cof='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs'
echo "ID experiment: $1"
echo "K initial: $2"
echo "K final: $3"

activation_d=160
ID=$1
gamma=0.1
lr=0.002
emb=256
batch=128
dataset='GLOBES'
arch='CAE224'
n_ep=100
n_ep_d=200
d_lab=0.002
str_ids='0,1,2,3'
# K parameters : This part guarantees the training of the model respect different number of clusters,
k_initial=$2
k_final=$3

# Train HERE YOU RUN YOUR PROGRAM

python train.py --id_exp ID${ID} --phase pretrain --embedded_dimension=${emb} --dataset_name=${dataset} --n_epochs=50 --n_epochs_decay=50 --AE_type=${arch} --save_latest_freq=5000 --gpu_ids=${str_ids} --verbose --dataset_name=${dataset} --config_dir=${cof} --reports_dir=${rep}

python train.py --id_exp ID${ID} --phase train --activation_delta=${activation_d} --batch_size=${batch} --n_epochs=${n_ep} --n_epochs_decay=${n_ep_d} --gamma=${gamma} --lr_tr=${lr} --delta_check --k_0=${k_initial} --k_fin=${k_final} --update_interval=2500 --delta_label=${d_lab} --embedded_dimension=${emb} --AE_type=${arch} --save_latest_freq=10000 --gpu_ids=${str_ids --verbose --dataset_name=${dataset} --config_dir=${cof} --reports_dir=${rep}
# Deactivate venv
deactivate