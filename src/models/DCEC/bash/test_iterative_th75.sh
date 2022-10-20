#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=V100:4
#SBATCH -t 0-18:00:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=filruf1702@gmail.com



# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/envs/venv
source bin/activate

# Load modules
module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0

# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/src/models/DCEC
pip install matplotlib
# Train HERE YOU RUN YOUR PROGRAM
python test.py --phase test --dataset_name CLARO --threshold=75 --k_0=4 --k_fin=20 --data_dir '/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data' --config_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs' --reports_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports' --embedded_dimension=256 --AE_type CAE512 --n_epochs=500 --gpu_ids '0,1,2,3'

# Deactivate venv
deactivate