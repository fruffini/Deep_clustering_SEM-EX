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


def iterative_evaluation_test():
    """ Function to evaluate every metric for each clustering method """
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
        fieldnames=['K', 'Var', 'Var_w', 'Gini']
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

    for k in np.arange(opt.k_0, opt.k_fin + 1):  # outer loop for different model instanced with different cluster number intialization MODEL_k -> MODEL_k+1
        #  _______________________________________________________________________________________________
        print(f"\n _______________________________________________________________________________________________ "
              f"\n INFORMATION: the current number of clusters is {k} "
              f"\n _______________________________________________________________________________________________")
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
            dict_out = model.compute_encoded(dataloader)
            x_out = dict_out['x_out']
            ids_tot = dict_out['id']
            z_latent = dict_out['z_latent']
            labels, q_ = model.get_predictions_probabilities(z_latent=z_latent)
            id_unique_dict, inverse_id_dict = util_data.find_unique_id_dictionary(ids_=ids_tot)

            # Metrics Computation
            computed_metrics = util_clustering.metrics_unsupervised_CVI(Z_latent_samples=z_latent, labels_clusters=labels)

            Si_score = computed_metrics['avg_Si_score']
            CH_score = computed_metrics['Calinski-Harabasz score']
            DB_score = computed_metrics['Davies-Douldin score']


            soft_label_mean_assegnation_score, avarage_P_prototypes, Mutual_information_score, indices_th, \
            P_for_cluster \
                = util_clustering.compute_probabilities_variables(
                labels=labels,
                ids=ids_tot,
                probability=q_,
                id_dict=id_unique_dict,
                threshold= opt.threshold
            )
            # Funzione di calcolo della varianza e della distribuzione delle slices dei pazienti nei clusters:
            Var_SF, Var_SF_W, list_Number_of_element, = util_clustering.TF_Variances_ECF(
                z_=z_latent,
                labels=labels,
                ids=ids_tot
                )
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
                    Var=Var_SF,
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

    iterative_evaluation_test()







