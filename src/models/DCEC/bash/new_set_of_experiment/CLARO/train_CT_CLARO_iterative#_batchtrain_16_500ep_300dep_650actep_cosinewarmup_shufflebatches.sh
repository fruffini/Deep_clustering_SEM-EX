#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:2
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


dat='/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data'
rep='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports'
cof='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs'
workdir='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX'
dataset='CLARO'
echo "ID experiment: $1"
echo "K initial: $2"
echo "K final: $3"
echo "Dataset name: $dataset"
activation_d=650
ID=$1


# Net configurations
dataset='CLARO'
arch='CAE256'

# Parameters for training

emb=128
batch_size=32
gamma=0.05
lr_tr=0.0008
n_ep=400
n_ep_d=300
d_lab=0.002 # if the delta labels is under 2 °% we can stop the training, only after 150 epochs
str_ids='0,1' # we use only two GPUs
shuffle_interval=20
# ID string for the experiment
ID=$1
# K parameters : This part guarantees the training of the model respect different number of clusters,
k_initial=$2
k_final=$3

# Train HERE YOU RUN YOUR PROGRAM

python train.py --id_exp ID_${ID} --experiment_name experiments_stage_2 --lr_tr=${lr_tr} --shuffle_interval=${shuffle_interval} --phase train --activation_delta=${activation_d} --batch_size=${batch_size} --n_epochs=${n_ep} --n_epochs_decay=${n_ep_d} --gamma=${gamma} --delta_check --k_0=${k_initial} --k_fin=${k_final} --update_interval=4500 --delta_label=${d_lab} --embedded_dimension=${emb} --box_apply --AE_type=${arch} --save_latest_freq=10000 --gpu_ids=${str_ids} --dataset_name=${dataset} --config_dir=${cof} --workdir=${workdir} --reports_dir=${rep} --data_dir=${dat} --verbose --delta_check

# Deactivate venv
deactivate