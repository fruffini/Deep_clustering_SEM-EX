# Test code for to avaluate
import os
import csv
import numpy as np
import torch

from util import util_clustering
from util import util_data
from util import util_general
from util import util_plots
from options.test_options import TestOptions
from util import util_path
from dataset import create_dataset
from models import create_model

global model
global dataset


def Plotter():
    """ Function to plot metrics once evaluation/encoding phase is completed """
    # ------ ITERATIVE TESTING OVER k ------
    # Training for k_num of clusters variable through list values of parameters' list.
    # k parameters values

    # initialize path manager for test path
    opt.path_man.initialize_test_folders()

    # paths for the metrics' files
    tables_dir = opt.path_man.get_path('tables_dir')
    plots_dir = opt.path_man.get_path('plots_dir')

    metrics_file_path = os.path.join(tables_dir, 'DCECs_log_clustering_metrics_over_k.csv')
    Var_gini_file_path = os.path.join(tables_dir, 'DCECs_log_variances_gini_.csv')
    probabilities_file_path = os.path.join(tables_dir, 'DCECs_log_probabilities_over_k.csv')

    util_plots.plot_metrics_unsupervised_K(
        file=metrics_file_path,
        save_dir=plots_dir
    )
    util_plots.plt_Var_Gini_K(
            file=Var_gini_file_path,
            save_dir=plots_dir
    )
    util_plots.plt_probabilities_NMI(
            file=probabilities_file_path,
            save_dir=plots_dir
    )


if __name__ == '__main__':
    import sys

    """sys.argv.extend(
        [
            '--phase', 'test',
            '--AE_type', 'CAE3',
            '--dataset_name', 'MNIST',
            '--reports_dir', 'C:\\Users\\Ruffi\\Desktop\\Deep_clustering_SEM-EX\\reports',
            '--config_dir', 'C:\\Users\\Ruffi\\Desktop\\Deep_clustering_SEM-EX\\configs',
        ]
    )"""
    Option = TestOptions() # test options
    opt =Option.parse()
    # Experiment Options

    opt.img_shape = (512, 512) if opt.dataset_name == "CLARO" else (28, 28)
    #  _______________________________________________________________________________________________
    # Submit run:
    print("Submit run")
    log_path = os.path.join(opt.reports_dir, 'log_run')
    run_id = util_path.get_next_run_id_local(os.path.join(log_path, opt.dataset_name), opt.phase)  # GET run id
    run_name = "{0:05d}--{1}--EXP_{2}".format(run_id, opt.phase, opt.id_exp)
    log_dir_exp = os.path.join(log_path, opt.dataset_name, run_name)
    util_general.mkdir(log_dir_exp)
    # Initialize Logger - run folder
    Option.print_options(opt=opt, path_log_run=log_dir_exp)
    logger = util_general.Logger(file_name=os.path.join(log_dir_exp, 'log.txt'), file_mode="w", should_flush=True)

    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    #                   DATA / MODEL / TRAIN
    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    # Welcome
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    print("Hello!", date_time)
    print("Running path for the experiment:", os.getcwd())
    # Info CUDA
    print("-------------------------INFO CUDA GPUs ----------------------------------")
    print("Number of GPUs devices: ", torch.cuda.device_count())
    print('CUDNN VERSION:', torch.backends.cudnn.version())
    for device in range(torch.cuda.device_count()):
        print("----DEVICE NUMBER : {%d} ----" % (device))
        print('__CUDA Device Name:', torch.cuda.get_device_name(device))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(device).total_memory / 1e9)
    print("-------------------------   INFO END   -----------------------------------")
    #  _______________________________________________________________________________________________
    # Dataset Options
    # In this point dataset and dataloader are genereted for the next step

    dataset = create_dataset(opt)
    opt.dataset_size = dataset.__len__()

    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    #           ITERATIVE EVALUATION/ TEST
    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________

    Plotter()







