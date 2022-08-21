import os

import torch

import util_general
log_dir = '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs'
logger = util_general.Logger(file_name=os.path.join(log_dir, 'log.txt'), file_mode="w", should_flush=True)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
else:
    print("Qualcosa non va nelle GPUs presenti per il Job!")
