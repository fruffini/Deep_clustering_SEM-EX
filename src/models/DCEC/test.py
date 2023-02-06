# Test code for to avaluate
import os
import csv
import json
import numpy as np
from util import util_clustering
from util import util_data
from util import util_general
from util import util_plots
from util.util_path import get_next_run_id_local
from options.test_options import TestOptions
from util import util_path
from dataset import *
from models import create_model
import sys
global model
global dataset
import pandas as pd

def iterative_evaluation_test():
    """ Function to evaluate every metric for each clustering method """
    # ------ ITERATIVE TESTING OVER k ------
    # Training for k_num of clusters variable through list values of parameters' list.
    # k parameters values

    # initialize path manager for test path
    global CDCC_
    opt.path_man.initialize_test_folders()

    # paths for the metrics' files
    save_dir = opt.path_man.get_path('save_dir')
    print('save_dir: ', save_dir)
    # Submit run:
    print("metrics_file")
    log_path_test = os.path.join(save_dir, 'test_run')
    run_id = get_next_run_id_local(log_path_test, opt.phase)  # GET run id
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d")
    run_name = "{0:05d}--{1}".format(run_id, date_time)
    log_dir_exp = os.path.join(log_path_test, run_name)
    util_general.mkdir(log_dir_exp)
    print('log_dir_exp', log_dir_exp)
    tables_dir = os.path.join(log_dir_exp, 'tables')
    plots_dir = os.path.join(log_dir_exp, 'plots')
    util_general.mkdirs([tables_dir, plots_dir])
    metrics_file_path = os.path.join(tables_dir, 'DCECs_log_clustering_metrics_over_k.csv')
    Var_gini_file_path = os.path.join(tables_dir, 'DCECs_log_variances_gini_.csv')
    probabilities_file_path = os.path.join(tables_dir, 'DCECs_log_probabilities_over_k.csv')
    Var_new_metrics = os.path.join(tables_dir, 'DCECs_new_metrics.csv')
    # CSV METRICS FILES :

    logfile_metrics_end = open(metrics_file_path, 'w')

    logfile_variances_Gini_csv = open(Var_gini_file_path, 'w')

    logfile_probabilities_csv = open(probabilities_file_path, 'w')

    # CSV Writers for log files :

    logwriter_metrics_unsup_end = csv.DictWriter(
        logfile_metrics_end,
        fieldnames=['K', 'SI_', 'CH_', 'DB_']
    )

    logwriter_variances_gini_end = csv.DictWriter(
        logfile_variances_Gini_csv,
        fieldnames=['K', 'Var_w', 'Gini']
    )

    logwriter_probabilities_end = csv.DictWriter(
        logfile_probabilities_csv,
        fieldnames=['K', 'mean_max_probabilities', 'mean_n_prototypes', 'Mutual_Information_Score', 'P_for_cluster']
    )

    # Scrittura header dei file csv:
    logwriter_metrics_unsup_end.writeheader()
    logwriter_probabilities_end.writeheader()
    logwriter_variances_gini_end.writeheader()
    dataloader = dataset.dataloader



    # N
    from easydict import EasyDict as edict
    new_metrics = edict({'VAR_w': list(), 'log_VAR_w': list(), 'VAR_w_100': list()})
    Separated_Data = edict()

    for k in np.arange(opt.k_0, opt.k_fin + 1):  # outer loop for different model instanced with different cluster number intialization MODEL_k -> MODEL_k+1
        #  _______________________________________________________________________________________________
        print(
            f"\n _______________________________________________________________________________________________ "
            f"\n INFORMATION: the current number of clusters is {k} "
            f"\n _______________________________________________________________________________________________"
        )
        #  _______________________________________________________________________________________________
        # Model selection and setup
        opt.num_clusters = k  # set the number of clusters.
        model = create_model(opt=opt)
        model.setup(opt=opt)
        if model.load_model_trained():
            print(
                f"\n _______________________________________________________________________________________________ "
                f"\n INFORMATION: the DCEC model with K = {k} has been loaded.  "
                f"\n _______________________________________________________________________________________________"
            )
            # Compute encoded samples
            labels_dir = os.path.join(model.load_dir, 'labels')
            labels_file = os.path.join(labels_dir, 'data_clusters_labels_K_{0}_.xlsx'.format(k))
            # load labels information
            if not os.path.exists(labels_file) and k == 16 and k == 17:
                labels_file = os.path.join(labels_dir, 'data_clusters_labels_K_{0}_.csv'.format(k))
                labels_info = pd.read_csv(labels_file)
            else:
                labels_info = pd.read_excel(labels_file)
            compressed_encoded_file = os.path.join(labels_dir, 'datasets_z_q_K_{}.npz'.format(k))




            # Sort the dataframe by column 'A'
            sorted_df = labels_info.sort_values(by='indexes')

            # Obtain the ordering index using argsort
            ordering = np.argsort(labels_info['indexes'])

            # Use the ordering index to sort another vector
            datasets_z_q = np.load(compressed_encoded_file)

            Z_encoded = datasets_z_q['Z_dataset']
            Q_encoded = datasets_z_q['Q_Dataset']

            # sort other vectors loaded inside this script file

            z_latent = Z_encoded[ordering]
            q_ = Q_encoded[ordering]
            labels = np.array(sorted_df['clusters_labels'])
            ids_tot = np.array(sorted_df['patient ID'])
            # -----------------------------------------------------------------------------------------------
            id_unique_dict, inverse_id_dict = util_data.find_unique_id_dictionary(ids_=ids_tot)
            # -----------------------------------------------------------------------------------------------
            plots_dir_labels_images = os.path.join(plots_dir, 'Clusters_images_{0}'.format(opt.num_clusters))
            # -----------------------------------------------------------------------------------------------
            if os.path.exists(plots_dir_labels_images):
                util_general.del_dir(plots_dir_labels_images)
            util_general.mkdir(plots_dir_labels_images)
            # Plot Clusters Examples

            # Create the dataset separated by cluster label
            Dataset_k = separate_dataset_by_label(opt, labels_info)
            clustering_K = k
            Separated_Data[f'K_{clustering_K}'] = Dataset_k
            for Dataset_label, Dataset in Dataset_k.get_labels_dict().items():
                util_plots.plot_label_examples(Dataset, clustering_k=clustering_K, save_dir=plots_dir_labels_images, Label_selected=Dataset_label.split('_')[-1], number_to_plot=12)






            # Metrics Computation
            computed_metrics = util_clustering.metrics_unsupervised_CVI(Z_latent_samples=z_latent, labels_clusters=labels)

            Si_score = computed_metrics['avg_Si_score']
            CH_score = computed_metrics['Calinski-Harabasz score']
            DB_score = computed_metrics['Davies-Douldin score']

            soft_label_mean_assegnation_score, avarage_P_prototypes, Mutual_information_score, indices_th, P_for_cluster  = util_clustering.compute_probabilities_variables(
                labels=labels,
                ids=ids_tot,
                probability=q_,
                id_dict=id_unique_dict,
                threshold=opt.threshold
            )
            # Funzione di calcolo della varianza e della distribuzione delle slices dei pazienti nei clusters:
            log_Var_SF_W, Var_SF_W, Var_SF_100, list_Number_of_element, = util_clustering.TF_Variances_ECF(
                z_=z_latent,
                labels=labels,
                ids=ids_tot
            )
            # first of all try a new alternative to the only variances metric
            new_metrics.VAR_w.append(Var_SF_W)
            new_metrics.log_VAR_w.append(log_Var_SF_W)
            new_metrics.VAR_w_100.append(Var_SF_100)
            if CDCC_ is None:
                CDCC_ = util_clustering.Metrics_CCDC(
                    opt=opt
                )
            CDCC_.add_new_Clustering_configuration(labels_info=sorted_df)

            # funzione di calcolo degli indici di Gini, Gini cumulato over k e dervata discreta di Gina na cert
            gini_index_over_t, gini_cumulative = util_clustering.compute_GINI(list_distribution=list_Number_of_element)

            # WRITE LINES IN THE CSV FILES

            logwriter_metrics_unsup_end.writerow(
                dict(
                    K=opt.num_clusters,
                    SI_=Si_score,
                    CH_=CH_score,
                    DB_=DB_score
                )
            )

            logwriter_variances_gini_end.writerow(
                dict(
                    K=opt.num_clusters,
                    Var_w=Var_SF_W,
                    Gini=gini_index_over_t
                )
            )
            logwriter_probabilities_end.writerow(
                dict(
                    K=opt.num_clusters,
                    mean_max_probabilities=soft_label_mean_assegnation_score,
                    mean_n_prototypes=avarage_P_prototypes,
                    Mutual_Information_Score=Mutual_information_score,
                    P_for_cluster=P_for_cluster
                )
            )



        else:
            raise NotImplementedError(" Pretrained weights not implemented, please launch experiment with --phase <pretrain>")

        #  _______________________________________________________________________________________________
        #  _______________________________________________________________________________________________
        # _____________________________________________________________________________________________________________________
        # END ITER K:
    logfile_variances_Gini_csv.close(), logfile_probabilities_csv.close(), logfile_metrics_end.close(),
    DICE_IOU_NMI_path = os.path.join(plots_dir, 'DICE_IOU_NMI')
    util_general.mkdir(os.path.join(plots_dir, 'DICE_IOU_NMI'))

    DICE_similarity_matrix, IOU_similarity_matrix, NMI_matrix, Adj_NMI_matrix, Cluster_Dimensions = CDCC_.compute_CCDC()

    encoded_row = {line[1].name: np.count_nonzero(line[1].where(line[1] > 0.05, 0)) for line in DICE_similarity_matrix.iterrows()}

    dict_ = {val: float() for val in np.arange(opt.k_0, opt.k_fin)}
    for key, value in encoded_row.items():
        dict_[int(key.split('_')[1])] += value
    # Save Data
    with open(Var_new_metrics, 'w') as file_metrics:
        json.dump(new_metrics, file_metrics)
        file_metrics.close()

    # Matrix Savings
    DICE_similarity_matrix.to_excel(os.path.join(DICE_IOU_NMI_path, f'DICE_matrix_{opt.dataset_name}.xlsx'))
    IOU_similarity_matrix.to_excel(os.path.join(DICE_IOU_NMI_path, f'IOU_matrix_{opt.dataset_name}.xlsx'))
    NMI_matrix.to_excel(os.path.join(DICE_IOU_NMI_path, f'NMI_matrix_{opt.dataset_name}.xlsx'))
    Adj_NMI_matrix.to_excel(os.path.join(DICE_IOU_NMI_path, f'Adj_NMI_matrix_{opt.dataset_name}.xlsx'))
    Cluster_Dimensions.to_excel(os.path.join(DICE_IOU_NMI_path, f'Cluster_Dims_matrix_{opt.dataset_name}.xlsx'))
    # saving directories and plots

    util_plots.plot_informations_over_clusters(
        data=NMI_matrix,
        opt=opt,
        save_dir=DICE_IOU_NMI_path,
        head='NMI'
    )
    util_plots.plot_informations_over_clusters(
        data=Adj_NMI_matrix,
        opt=opt,
        save_dir=DICE_IOU_NMI_path,
        head='Adj_NMI'
    )
    util_plots.plot_metrics_unsupervised_K(
        file=metrics_file_path,
        save_dir=plots_dir
    )
    util_plots.plt_Var_Gini_K(
        file=Var_gini_file_path,
        save_dir=plots_dir
    )
    util_plots.plt_Var_new_metrics(
        file=Var_new_metrics,
        save_dir=plots_dir,
        opt=opt
    )
    """
    util_plots.plt_probabilities_NMI(
        file=probabilities_file_path,
        save_dir=plots_dir
    )
    """


# --------------------------------------------------------------------------------------------------
# OPENING FUNCTION CALL: RUN/DEBUG

def debugging_only():
    print("".center(100, '°'))
    print(" DEBUG MODALITY ".center(100, '°'))
    print("".center(100, '°'))
    '''
    sys.argv.extend(
            [   '--phase', 'test',
                '--dataset_name', 'CLARO',
                
                '--experiment_name', 'experiments_stage_2',
                '--reports_dir', '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports',
                '--config_dir', '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs',
                '--embedded_dimension', '128',
                '--AE_type', 'CAE256',
                '--gpu_ids','0',
                '--id_exp','ID_1',
                '--threshold', '95',
                '--k_0', '3',
                '--k_fin', '14',
                '--num_threads', '1',
            ]
        )
    '''
    sys.argv.extend(
            [   '--phase', 'test',
                '--dataset_name', 'MNIST',
                '--experiment_name', 'experiments_stage_2',
                '--reports_dir', r'C:\Users\Ruffi\Desktop\Deep_clustering_SEM-EX\reports',
                '--config_dir', r'C:\Users\Ruffi\Desktop\Deep_clustering_SEM-EX\configs',
                '--embedded_dimension', '10',
                '--AE_type', 'CAEMNIST',
                '--gpu_ids','0',
                '--id_exp','ID_1',
                '--threshold', '95',
                '--k_0', '3',
                '--k_fin', '10',
                '--num_threads', '1',
            ]
        )

def running():
    print("".center(100, '*'))
    print(" RUNNING MODALITY ".center(100, '*'))
    print("".center(100, '*'))

if __name__ == '__main__':
    """
    Detecting if you're in the PyCharm debugger or not
    If you put a breakpoint HERE and look at the callstack you will 
    see the entry point is in 'pydevd.py'
    In debug mode, it copies off sys.argv: "sys.original_argv = sys.argv[:]"
    We abuse this knowledge to test for the PyCharm debugger.
    """

    if util_general.is_debug():
        running = debugging_only
    running()

    util_general.print_CUDA_info()
    Option = TestOptions()  # test options
    opt = Option.parse()
    # Experiment Options
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

    # Welcome

    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    print("Hello!", date_time)
    print("Running path for the experiment:", os.getcwd())
    util_general.print_CUDA_info()
    # Dataset Options
    # In this point dataset and dataloader are genereted for the next step
    dataset = create_dataset(opt)
    opt.dataset_size = dataset.__len__()
    CDCC_ = None
    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    #           ITERATIVE EVALUATION/ TEST
    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    try:
        iterative_evaluation_test()
        # Raising KeyboardInterruption
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("Program terminated manually!")
        raise SystemExit







