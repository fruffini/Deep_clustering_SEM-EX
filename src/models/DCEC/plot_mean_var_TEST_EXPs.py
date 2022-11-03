# Test code for to avaluate
import os
import csv

import click
import numpy as np
import torch
from easydict import EasyDict

from util import util_clustering
from util import util_data
from util import util_general
from util import util_plots
from options.test_options import TestOptions
from util import util_path
from dataset import create_dataset
from models import create_model
from datetime import datetime
global model
global dataset
import sys
class ConvertStrToList(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            value = str(value)
            assert value.count('[') == 1 and value.count(']') == 1
            return list(int(x) for x in value.replace('"', "'").split('[')[1].split(']')[0].split(','))
        except Exception:
            raise click.BadParameter(value)
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
    now = datetime.now()
    date_time = now.strftime("[(%d/%m/%Y)(%H:%M:%S)]")
    opt.path_man.set_dir(dir_to_extend='plots_dir', name_att="new_plots", path_ext=f"Plots__{date_time}")
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
sys.argv.extend(
        [
            '--exp_date', 'last',
            '--id_interval', '[1,3]'

        ]
    )
@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--data_dir', help='Directory for input dataset', default='/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data', metavar='PATH')
@click.option('--phase', help='Phase.',  type=str, default='test')
@click.option('--reports_dir', help='Directory for reports', default='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports', metavar='PATH')
@click.option('--dataset_name', help='Name of the input dataset',  type=str, default='CLARO')
@click.option('--max_patients', help='Number of patients to preprocess', type=int, default=100000)
@click.option('--exp_date', help='Date of the experiment to compact in a single plot', type=click.Choice(['last']), required=True)
@click.option('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
@click.option('--id_interval', cls=ConvertStrToList, default=[],  help='Exp ID list to test and evaluate. ')
def main(**kwargs):

    opt = EasyDict(**kwargs)

    # Configuration Functions:
    # Submit run:
    print("Submit run")
    log_path = os.path.join(opt.reports_dir, '_MultiExps_log_run')
    run_id = util_path.get_next_run_id_local(os.path.join(log_path, opt.dataset_name), opt.phase)
    now = datetime.now()
    date_time = now.strftime("%d:%m:%Y")
    # GET run id
    run_name = "{0:05d}--{1}".format(run_id, date_time)
    log_dir_exp = os.path.join(log_path, opt.dataset_name, run_name)
    util_general.mkdir(log_dir_exp)
    # Initialize Logger - run folder
    logger = util_general.Logger(file_name=os.path.join(log_dir_exp, 'log.txt'), file_mode="w", should_flush=True)

    # Welcome
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
    #  _______________________________________________________________________________________________
    #           ITERATIVE EVALUATION MULTIEXPs
    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    for ID in range(opt.id_interval[0], opt.id_interval[1] + 1):

        path_exp = '/ mimer / NOBACKUP / groups / snic2022 - 5 - 277 / fruffini / SEM - EX / reports / experiment_name_CLARO / test / EXP_ID{0}'.format(ID).replace(' ', '')
        path_log_run = os.path.join(path_exp, 'test_run',opt.dataset_name)
        util_path.find_exp(path_log_run=path_log_run, MODE='last')







    Plotter()
if __name__ == '__main__':
    main()






