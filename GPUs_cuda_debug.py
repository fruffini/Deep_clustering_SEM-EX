import os

import torch

from src.models.DCEC.util import util_general
#log_dir = '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs'
log_dir = 'C:/Users/Ruffi/Desktop/Deep_clustering_SEM-EX'
logger = util_general.Logger(file_name=os.path.join(log_dir, 'log_cuda_info.txt'), file_mode="w", should_flush=True)
from datetime import datetime
now = datetime.now()
date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
print("Hello!", date_time)

print("----INFO CUDA GPUs ----")
print("Number of GPUs devices: ", torch.cuda.device_count())
print('__CUDNN VERSION:', torch.backends.cudnn.version())

for device in range(torch.cuda.device_count()):
    print("----DEVICE NUMBER : {%d} ----" % (device))
    print('__CUDA Device Name:', torch.cuda.get_device_name(device))
    print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(device).total_memory / 1e9)

