#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:3
#SBATCH -t 2-00:00:00
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
echo "ID experiment: $1"
echo "K initial: $2"
echo "K final: $3"

activation_d=70
ID=$1
gamma=0.03
lr=0.001
emb=256
batch=64
arch='CAE224'
n_ep=100
n_ep_d=100
d_lab=0.002
str_ids='0,1,2'
# K parameters : This part guarantees the training of the model respect different number of clusters,
ID=$1
k_initial=$2
k_final=$3
dataset=$4
# Train HERE YOU RUN YOUR PROGRAM

python train.py --id_exp ID_${dataset}_${ID} --lr_policy cosine-warmup --phase train --activation_delta=${activation_d} --batch_size=${batch} --n_epochs=${n_ep} --n_epochs_decay=${n_ep_d} --gamma=${gamma} --lr_tr=${lr} --delta_check --k_0=${k_initial} --k_fin=${k_final} --update_interval=2500 --delta_label=${d_lab} --embedded_dimension=${emb} --AE_type=${arch} --save_latest_freq=10000 --gpu_ids=${str_ids} --verbose --dataset_name=${dataset} --config_dir=${cof} --reports_dir=${rep} --shuffle_batches

# Deactivate venv
deactivate