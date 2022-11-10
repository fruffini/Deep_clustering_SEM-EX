#!/usr/bin/env bash


#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=V100:4
#SBATCH -t 3-1:00:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it

# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/envs/venv
source bin/activate

# Load modules
module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0

# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/src/models/DCEC

rep='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports'
dat='/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data'
cof='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs'
echo "ID experiment: $1"
echo "K initial: $2"
echo "K final: $3"


ID=$1
gamma=0.07
lr=0.0002
emb=256
batch=128
dataset='CLARO'
arch='CAE224'
n_ep=500
n_ep_d=300
d_lab=0.002
str_ids='0,1,2,3'
# K parameters : This part guarantees the training of the model respect different number of clusters,
k_initial=$2
k_final=$3

# Train HERE YOU RUN YOUR PROGRAM
python train.py --id_exp ID${ID} --phase train --batch_size=${batch} --n_epochs=${n_ep} --n_epochs_decay=${n_ep_d} --gamma=${gamma} --lr_tr=${lr} --delta_check --k_0=${k_initial} --k_fin=${k_final} --update_interval=3500 --delta_label=${d_lab} --embedded_dimension=${emb} --AE_type=${arch} --save_latest_freq=14000 --gpu_ids=${str_ids} --box_apply --verbose --dataset_name=${dataset} --data_dir=${dat} --config_dir=${cof} --reports_dir=${rep}
# Deactivate venv
deactivate