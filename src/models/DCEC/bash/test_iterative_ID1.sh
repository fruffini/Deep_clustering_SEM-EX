#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:2
#SBATCH -t 0-18:00:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=filruf1702@gmail.com



# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/envs/SEM-EX-env
source bin/activate

# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/src/models/DCEC
pip install matplotlib
# Train HERE YOU RUN YOUR PROGRAM
python test.py --phase test --dataset_name CLARO --threshold=95 --k_0=3 --k_fin=15 --data_dir '/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data' --config_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs' --reports_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports' --embedded_dimension=256 --AE_type CAE224 --gpu_ids '0,1' --id_exp ID1 --box_apply

# Deactivate venv
deactivate