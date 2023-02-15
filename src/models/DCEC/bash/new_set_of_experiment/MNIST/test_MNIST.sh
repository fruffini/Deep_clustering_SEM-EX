#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 0-01:00:00
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
dataset='MNIST'
echo "ID experiment: $1"
echo "K initial: $2"
echo "K final: $3"
echo "Dataset name: $dataset"
ID=$1


# Net configurations
dataset='MNIST'
arch='CAEMNIST'
phase='test'
# Parameters for training
emb=10
str_ids='0' # we use only two GPUs
# ID string for the experiment
ID=$1
# K parameters : This part guarantees the training of the model respect different number of clusters,
k_initial=$2
k_final=$3
exp_name='experiments_stage_2'
# Train HERE YOU RUN YOUR PROGRAM

python test.py --id_exp ID_${ID} --experiment_name=${exp_name} --phase=${phase} --k_0=${k_initial} --k_fin=${k_final} --embedded_dimension=${emb} --AE_type=${arch} --gpu_ids=${str_ids} --dataset_name=${dataset} --config_dir=${cof} --workdir=${workdir} --reports_dir=${rep} --verbose

deactivate