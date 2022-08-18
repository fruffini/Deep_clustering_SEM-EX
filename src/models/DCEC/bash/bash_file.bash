#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH -t 0-02:00:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=filruf1702@gmail.com

# Load modules
module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0

# Activate venv
cd /mimer/NOBACKUP/groups/inphai/vallenv
source bin/activate

# Executes the code 
cd /mimer/NOBACKUP/groups/snic2022-5-277/mpastina/Brain_XAI

# Train HERE YOU RUN YOUR PROGRAM
python3 ./Testing_MLP.py

# Deactivate venv
deactivate