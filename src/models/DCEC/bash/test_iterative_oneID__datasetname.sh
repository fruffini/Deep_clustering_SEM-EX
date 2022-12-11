#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=T4:2
#SBATCH -t 0-3:00:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it


# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/envs/SEM-EX-env
source bin/activate



# Load modules
module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0



# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/src/models/DCEC
# Train HERE YOU RUN YOUR PROGRAM
dataset=$1
if [ dataset = 'CLARO' ]
then
  dat='/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data'
  pwd
fi
dat='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/src/models/DCEC/data'
rep='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports'
cof='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs'
echo "ID experiment: $2"
echo "K initial: $3"
echo "K final: $4"

ID=$2
emb=256
arch='CAE224'
th=95
str_ids='0,1'
# K parameters : This part guarantees the training of the model respect different number of clusters,
k_initial=$3
k_final=$4


python test.py --id_exp ID${ID} --redo_encoding --phase test --threshold=${th} --k_0=${k_initial} --k_fin=${k_final} --gpu_ids=${str_ids} --embedded_dimension=${emb} --AE_type=${arch} --gpu_ids=${str_ids} --verbose --dataset_name=${dataset} --data_dir=${dat} --config_dir=${cof} --reports_dir=${rep}
# Deactivate venv
deactivate