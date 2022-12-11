# Test code for to avaluate
global dataset
import os
import csv
import click
import numpy as np
import torch
from easydict import EasyDict
from util import util_data
from util import util_general
from util import util_plots
from util import util_path
from datetime import datetime
global model
import sys

"""def Plotter():
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
    """
sys.argv.extend(
        [
            '--exp_mode', 'last',
            '--id_interval', '[1,5]'

        ]
    )
@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--data_dir', help='Directory for input dataset', default='/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data', metavar='PATH')
@click.option('--phase', help='Phase.',  type=str, default='test')
@click.option('--reports_dir', help='Directory for reports', default='/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports', metavar='PATH')
@click.option('--dataset_name', help='Name of the input dataset',  type=str, default='CLARO')
@click.option('--exp_mode', help='Date of the experiment to compact in a single plot', type=click.Choice(['last']), required=True)
@click.option('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
@click.option('--id_interval', cls=util_general.ConvertStrToList, default=[],  help='Exp ID list to test and evaluate. ')
def main(**kwargs):

    opt = EasyDict(**kwargs)

    # Configuration Functions:
    # Submit run:
    print("Submit run")
    log_path = os.path.join(opt.reports_dir, '_MultiExps_log_run')
    run_id = util_path.generate_id_for_multi_exps(os.path.join(log_path, opt.dataset_name))
    now = datetime.now()
    date_time = now.strftime("%d:%m:%Y")
    # GET run id
    run_name = "{0:05d}--{1}".format(run_id, date_time)
    log_dir_exp = os.path.join(log_path, opt.dataset_name, run_name)
    tables_dir = os.path.join(log_dir_exp, 'tables')
    plots_dir = os.path.join(log_dir_exp, 'plots')
    util_general.mkdirs([tables_dir, plots_dir])
    tables_dir_exps = tables_dir
    plots_dir_exps = plots_dir
    # Initialize Logger - run folder
    logger = util_general.Logger(file_name=os.path.join(log_dir_exp, 'log.txt'), file_mode="w", should_flush=True)

    metrics_file_path = os.path.join(tables_dir_exps, 'metrics_over_EXPs.csv')
    Var_gini_file_path = os.path.join(tables_dir_exps, 'variances_gini_over_EXPs.csv')
    probabilities_file_path = os.path.join(tables_dir_exps, 'probabilities_over_EXPs.csv')
    # CSV METRICS FILES :

    logfile_metrics_end = open(metrics_file_path, 'w')

    logfile_variances_Gini_csv = open(Var_gini_file_path, 'w')

    logfile_probabilities_csv = open(probabilities_file_path, 'w')

    # CSV Writers for log files :

    logwriter_UL = csv.DictWriter(
        logfile_metrics_end,
        fieldnames=[
            'K',
            'SI__mean',
            'CH__mean',
            'DB__mean',
            'SI__var',
            'CH__var',
            'DB__var'
        ]

    )

    logwriter_VG = csv.DictWriter(
        logfile_variances_Gini_csv,
        fieldnames=[
            'K',
            'Var_mean',
            'Var_w_mean',
            'Gini_mean',
            'Var_var',
            'Var_w_var',
            'Gini_var']
    )

    logwriter_Pb = csv.DictWriter(
        logfile_probabilities_csv,
        fieldnames=[
            'K', 'mean_max_probabilities_mean',
            'mean_n_prototypes_mean',
            'Mutual_Information_Score_mean',
            'P_for_cluster_mean',
            'mean_max_probabilities_var',
            'mean_n_prototypes_var',
            'Mutual_Information_Score_var',
            'P_for_cluster_var',
            'Metrics_Combined_mean',
            'Metrics_Combined_var'
            ]
    )

    # Scrittura header dei file csv:
    logwriter_UL.writeheader()
    logwriter_VG.writeheader()
    logwriter_Pb.writeheader()




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
    metrics_list_over_exps = list()
    probabilities_list_over_exps = list()
    Var_GINI_list_over_exps = list()


    for ID in range(opt.id_interval[0], opt.id_interval[1] + 1):

        path_exp = '/ mimer / NOBACKUP / groups / snic2022 - 5 - 277 / fruffini / SEM - EX / reports / experiment_name_CLARO / test / EXP_ID{0}'.format(ID).replace(' ', '')
        path_log_run = os.path.join(path_exp, 'test_run',opt.dataset_name)
        log_dir_exp = util_path.find_last_exp(path_log_run=path_log_run, MODE=opt.exp_mode)


        tables_dir = os.path.join(log_dir_exp, 'tables')
        plots_dir = os.path.join(log_dir_exp, 'plots')
        util_general.mkdirs([tables_dir, plots_dir])
        metrics_OK_file = os.path.join(tables_dir, 'DCECs_log_clustering_metrics_over_k.csv')
        Var_gini_OK_file = os.path.join(tables_dir, 'DCECs_log_variances_gini_.csv')
        probabilities_OK_file = os.path.join(tables_dir, 'DCECs_log_probabilities_over_k.csv')

        import pandas as pd
        metrics_list_over_exps.append(pd.read_csv(metrics_OK_file))

        prb = pd.read_csv(probabilities_OK_file)
        matrix_P = [list(map(float, i[1:-1].split(', '))) for i in prb["P_for_cluster"]]
        matrix_mean_P = [np.mean(list(map(float, i[1:-1].split(', ')))) for i in prb["P_for_cluster"]]
        matrix_NMI = np.array(prb["Mutual_Information_Score"])
        prb["P_for_cluster"] = matrix_mean_P
        Mutual_Information_Score = (matrix_NMI - np.min(matrix_NMI)) / (np.max(matrix_NMI) - np.min(matrix_NMI))
        mean_assegnation = (matrix_mean_P - np.min(matrix_mean_P)) / (np.max(matrix_mean_P) - np.min(matrix_mean_P))
        Metrics_Combined = np.sqrt(Mutual_Information_Score * mean_assegnation)
        prb["Metrics_Combined"] = Metrics_Combined
        probabilities_list_over_exps.append(prb)

        v_g = pd.read_csv(Var_gini_OK_file)
        matrix_gini = np.array([np.mean(tuple(map(float, i[1:-1].split(', ')))) for i in v_g["Gini"]])
        matrix_Var = np.array([np.mean(tuple(map(float, i[1:-1].split(', ')))) for i in v_g["Var"]])
        matrix_Var_w = np.array([np.mean(tuple(map(float, i[1:-1].split(', ')))) for i in v_g["Var_w"]])
        v_g["Gini"] = matrix_gini
        v_g["Var"] = matrix_Var
        v_g["Var_w"] = matrix_Var_w

        Var_GINI_list_over_exps.append(v_g)
        del v_g, matrix_Var, matrix_Var_w, matrix_gini, prb, Metrics_Combined, matrix_mean_P, Mutual_Information_Score


    UL_var_mean = util_data.mean_var_over_Exps(list_exps=metrics_list_over_exps, columns = metrics_list_over_exps[0].columns)
    P_var_mean = util_data.mean_var_over_Exps(list_exps=probabilities_list_over_exps, columns=probabilities_list_over_exps[0].columns)
    Var_G_var_mean = util_data.mean_var_over_Exps(list_exps=Var_GINI_list_over_exps, columns= Var_GINI_list_over_exps[0].columns)

    # WRITE LINES IN THE CSV FILES

    logwriter_UL.writerow(
        dict(
            K=15,
            **UL_var_mean
        )
    )

    logwriter_VG.writerow(
        dict(
            K=15,
            **Var_G_var_mean
        )
    )
    logwriter_Pb.writerow(
        dict(
            K=15,
            **P_var_mean
        )
    )

    logfile_metrics_end.close(), logfile_probabilities_csv.close(), logfile_variances_Gini_csv.close(),
    K = metrics_list_over_exps[0]['K']
    util_plots.plot_metrics_unsupervised_EXPs_mean_var(data=UL_var_mean, save_dir=plots_dir_exps, Ks=K)
    util_plots.plt_Var_Gini_K_EXPs_mean_var(data=Var_G_var_mean, save_dir=plots_dir_exps, Ks=K)
    util_plots.plt_probabilities_NMI_mean_var(data=P_var_mean, save_dir=plots_dir_exps, Ks=K)
    print('here')


















#    Plotter()
if __name__ == '__main__':
    main()






