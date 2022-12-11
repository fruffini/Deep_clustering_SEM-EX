#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:3
#SBATCH -t 2-10:00:00
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
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/envs/SEM-EX-env
source bin/activate



# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/src/models/DCEC

rep='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports'
cof='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs'
dat='/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data'
echo "ID experiment: $1"
echo "K initial: $2"
echo "K final: $3"

activation_d=600
###############################################################################################################
###############################################################################################################
gamma=0.03
###############################################################################################################
###############################################################################################################

lr=0.0002
emb=256
batch=128
dataset='CLARO'
arch='CAE224'
n_ep=500
n_ep_d=350
d_lab=0.002
str_ids='0,1,2'
# K parameters : This part guarantees the training of the model respect different number of clusters,


ID=$1
k_initial=$2
k_final=$3

# Train HERE YOU RUN YOUR PROGRAM


python train.py --id_exp ID${ID} --phase pretrain --embedded_dimension=${emb} --dataset_name=${dataset} --n_epochs=50 --n_epochs_decay=50 --AE_type=${arch} --save_latest_freq=5000 --gpu_ids=${str_ids} --verbose --box_apply --data_dir=${dat} --config_dir=${cof} --reports_dir=${rep}

python train.py --lr_policy cosine-warmup --shuffle_batches --id_exp ID${ID} --phase train --box_apply --activation_delta=${activation_d} --batch_size=${batch} --n_epochs=${n_ep} --n_epochs_decay=${n_ep_d} --gamma=${gamma} --lr_tr=${lr} --delta_check --k_0=${k_initial} --k_fin=${k_final} --update_interval=2500 --delta_label=${d_lab} --embedded_dimension=${emb} --AE_type=${arch} --save_latest_freq=10000 --gpu_ids=${str_ids} --verbose --dataset_name=${dataset} --config_dir=${cof} --reports_dir=${rep} --data_dir=${dat}

# Deactivate venv
deactivate



