#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:3
#SBATCH -t 3-50:00:00
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
workdir='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX'
dataset='GLOBES_2'
echo "ID experiment: $1"
echo "K initial: $2"
echo "K final: $3"
echo "Dataset name: $dataset"
activation_d=200
ID=$1


# Net configurations
arch='CAE224'
# Parameters for training
shuffle_interval=30
lr_policy='cosine-warmup'
emb=128
batch_size=64
gamma=0.05
lr_pr=0.001
lr_tr=0.00005
n_ep=150
n_ep_d=100
d_lab=0.002 # if the delta labels is under 2 °% we can stop the training, only after 150 epochs
str_ids='0,1,2' # we use only two GPUs

# ID string for the experiment
ID=$1
# K parameters : This part guarantees the training of the model respect different number of clusters,
k_initial=$2
k_final=$3
echo %J
# Train HERE YOU RUN YOUR PROGRAM

python train.py --id_exp ID_${ID} --experiment_name experiments_stage_2 --phase pretrain --lr_pr=${lr_pr} --embedded_dimension=${emb} --dataset_name=${dataset} --n_epochs=25 --n_epochs_decay=25 --AE_type=${arch} --save_latest_freq=5000 --gpu_ids=${str_ids} --dataset_name=${dataset} --config_dir=${cof} --workdir=${workdir} --reports_dir=${rep} --shuffle_batches  --verbose

python train.py --id_exp ID_${ID} --shuffle_interval=${shuffle_interval} --lr_policy=${lr_policy} --experiment_name experiments_stage_2 --lr_tr=${lr_tr} --phase train --activation_delta=${activation_d} --batch_size=${batch_size} --n_epochs=${n_ep} --n_epochs_decay=${n_ep_d} --gamma=${gamma} --delta_check --k_0=${k_initial} --k_fin=${k_final} --update_interval=4500 --delta_label=${d_lab} --embedded_dimension=${emb} --AE_type=${arch} --save_latest_freq=20000 --gpu_ids=${str_ids} --dataset_name=${dataset} --config_dir=${cof} --workdir=${workdir} --reports_dir=${rep} --shuffle_batches --verbose --delta_check

# Deactivate venv
deactivate