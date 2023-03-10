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
# shellcheck disable=SC2164
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/envs/venv
source bin/activate

# Load modules
module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0

# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/src/models/DCEC





# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/envs/venv
source bin/activate

# Load modules
module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0

# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/src/models/DCEC

# Train HERE YOU RUN YOUR PROGRAM
python train.py --phase pretrain --data_dir '/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data' --config_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs' --reports_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports' --embedded_dimension=256 --dataset_name CLARO --AE_type CAE224 --n_epochs=50 --n_epochs_decay=50 --save_latest_freq=5000 --gpu_ids '0,1,2,3' --verbose --id_exp ID1 --box_apply

# Train HERE YOU RUN YOUR PROGRAM
python train.py --batch_size=128 --gamma=0.07 --lr_tr=0.0002 --delta_check --phase train --dataset_name CLARO --k_0=3 --k_fin=15 --update_interval=3500 --delta_label=0.002 --data_dir '/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data' --config_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs' --reports_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports' --embedded_dimension=256 --AE_type CAE224 --n_epochs=500 --n_epochs_decay=300 --save_latest_freq=14000 --gpu_ids '0,1,2,3' --box_apply --verbose --id_exp ID1

# Deactivate venv
deactivate



outdir="/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/Gan-track/reports"
dataset="claro"
modalities="CT"
metric="fid50k_full"
# Clean results
exp_in= 1



for i in $(seq $ $END);
  do
    # Train HERE YOU RUN YOUR PROGRAM
    python train.py --phase pretrain --data_dir '/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data' --config_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs' --reports_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports' --embedded_dimension=256 --dataset_name CLARO --AE_type CAE224 --n_epochs=50 --n_epochs_decay=50 --save_latest_freq=5000 --gpu_ids '0,1,2,3' --verbose --id_exp ID${ID} --box_apply
    python train.py --phase train --batch_size=128 --gamma=0.07 --lr_tr=0.0002 --delta_check --dataset_name CLARO --k_0=3 --k_fin=15 --update_interval=3500 --delta_label=0.002 --data_dir '/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data' --config_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs' --reports_dir '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports' --embedded_dimension=256 --AE_type CAE224 --n_epochs=500 --n_epochs_decay=300 --save_latest_freq=14000 --gpu_ids '0,1,2,3' --box_apply --verbose --id_exp ID${ID}


  done

echo "done"