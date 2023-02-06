# Implementation of DEEP-CONVOLUTIONAL-EMBEDDED_CLUSTERING for semantic-extraction procedure
# defined in the method SEM-EX
# hello here
from __future__ import print_function, division
import sys
import time
import os
from datetime import datetime
import torch
from tqdm import tqdm
from util.util_path import get_next_run_id_local
from options.train_options import TrainOptions
from models import create_model
from util import util_general
from dataset import create_dataset


import numpy as np


def pretrain():
    # ----- PRETRAIN ------
    total_iters = 0  # the total number of training iterations
    opt.save_latest_freq = opt.batch_size * np.ceil(opt.save_latest_freq / opt.batch_size)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs
        # Train the model for each value inside the k_values option list
        epoch_iter = 0
        model.update_learning_rate() if not total_iters == 0 else model.do_nothing()  # update learning rates linked to optimizer,
        with tqdm(dataset.dataloader, unit="batch", desc="Progress bar pretraining phase") as tqdm_pret:
            for i, data in enumerate(tqdm_pret):
                tqdm_pret.set_description(f"Epoch {epoch}")
                # inner loop within one epoch
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)  # Unpack data from dataset in batch set and apply preprocessing
                model.optimize_parameters()
                # Accumulate all losses value
                model.accumulate_losses()  # accumulate losses in a dictionary
                if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('\nsaving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
                    model.save_image_reconstructed(epoch=epoch)
                    model.print_current_losses(epoch=epoch, iters=epoch_iter)
            model.reset_accumulator()


def train():
    # ----- TRAIN ------
    # starting procedure for training
    model.prepare_training(dataset=dataset)
    exit_ = False  # training exit condition
    total_iters = 0  # the total number of training iterations
    batch_iters = 0  # the total number of batch processed
    epoch_counter = 0
    updated_target = False
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs
        # Long Interval Between Epochs
        if epoch_counter % opt.shuffle_interval == 0 and opt.shuffle_interval != 0:
            model.save_image_reconstructed(epoch=epoch)
            dataset.shuffle_data()
            delta_bool = model.update_target(dataloader=dataset.dataloader_big_batch, indexing=dataset.get_new_indexig())
            updated_target = True
            epoch_counter = 0

        # Train the model for each value inside the k_values option list
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0  # the total number of training iterations during a single epoch
        model.update_learning_rate() if not total_iters == 0 else model.do_nothing()  # update learning rates linked to optimizers.

        for ind, data in enumerate(dataset.dataloader):
            # inner loop within one epoch
            if updated_target:
                model.print_metrics(epoch=epoch)
                updated_target = False
                if total_iters > 0 and delta_bool and not np.unique(model.y_prediction).__len__() == 1 and epoch > opt.activation_delta:
                    print('\n Reached tolerance threshold. Stopping training.\n', flush=False)
                    model.save_networks('early_stopped')
                    exit_ = True
                    break
                else:
                    print('\nTolerance on labels difference beetween %s iteration not respected. Continue Training...\n' %(opt.update_interval), flush=False)
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            batch_iters += 1
            model.set_input(data)  # Unpack data from dataset in q_ij set and apply preprocessing
            model.set_target_p_batch(ind=ind)
            model.optimize_parameters()
            # Accumulate all losses value
            model.accumulate_losses()  # accumulate losses in a dictionary

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters), )
                save_suffix = 'iter_%d_epoch_%d' % (total_iters, epoch) if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        model.print_current_losses(epoch=epoch, iters=epoch_iter)
        model.reset_accumulator()
        epoch_counter += 1
        print('Training time for 1 epoch : ', np.round((time.time() - epoch_start_time), 2), ' ( sec )')
        if exit_:
            model.save_networks('end_training')
            return
        else:
            continue
    model.save_networks('end_training')
    print('INFO: training ended!')


def iterative_training_over_k():
    # ------ ITERATIVE TRAINING OVER k ------
    # Training for k_num of clusters variable through list values of parameters' list.
    # k parameters values
    global model
    for k in np.arange(opt.k_0, opt.k_fin + 1):  # outer loop for different model instanced with different cluster number intialization MODEL_k -> MODEL_k+1
        #  _______________________________________________________________________________________________

        print("".center(100, '_'))
        print(f" INFORMATION: the current number of clusters is {k}  ".center(100, '_'))
        print("".center(100, '_'))

        #  _______________________________________________________________________________________________
        # Model selection and setup
        opt.num_clusters = k  # set the number of clusters.
        model = create_model(opt=opt)
        model.setup(opt=opt)
        if model.load_model_pretrained():
            train()  # train function.
            print("".center(100, '_'))
            print(" INFORMATION: the DCEC model with the number of clusters equal to {k} has been trained ".center(100, '_'))
            print("".center(100, '_'))
        else:
            raise NotImplementedError(" Pretrained weights not implemented, please launch experiment with --phase <pretrain>")

        #  _______________________________________________________________________________________________
        #  _______________________________________________________________________________________________


def debugging_only():
    """ This function is called only if in the DEBUG modality of Pycharm """

    print("".center(100, '°'))
    print(" DEBUG MODALITY ".center(100, '°'))
    print("".center(100, '°'))
    # local DEBUGGER
    if 'Ruffi' in sys.executable:
        # TRAIN
        sys.argv.extend(
            [
                '--workdir', r'C:\Users\Ruffi\Desktop\Deep_clustering_SEM-EX',
                '--phase', 'train',
                '--dataset_name', 'MNIST',
                '--perc', '0.1',
                '--mnist_mode', 'normal',
                '--lr_policy', 'cosine-warmup',
                '--reports_dir', r'C:\Users\Ruffi\Desktop\Deep_clustering_SEM-EX\reports',
                '--config_dir', r'C:\Users\Ruffi\Desktop\Deep_clustering_SEM-EX\configs',
                '--embedded_dimension', '10',
                '--AE_type', 'CAEMNIST',
                '--gpu_ids', '0',
                '--id_exp', 'ID_1',
                '--k_0', '3',
                '--k_fin', '14',
                '--mnist_mode', 'normal',
                '--n_epochs', '250',
                '--n_epochs_decay', '250',
                '--num_threads', '4',
                '--experiment_name', 'new_experiments',
                '--shuffle_interval', '1',
                '--verbose',
                '--delta_check',
                '--batch_size', '64',
                '--lr_tr', '0.001'
            ]
        )
    """
    sys.argv.extend(
        [
            '--workdir', '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX',
            '--phase', 'train',
            '--dataset_name', 'GLOBES_2',
            '--lr_policy', 'cosine-warmup',
            '--reports_dir','/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/reports',
            '--config_dir', '/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/SEM-EX/configs',
            '--embedded_dimension', '64',
            '--AE_type', 'CAE224',
            '--gpu_ids', '3',
            '--id_exp', 'ID_1',
            '--k_0', '2',
            '--k_fin', '15',
            '--n_epochs', '100',
            '--perc', '0.1',
            '--n_epochs_decay', '100',
            '--num_threads', '4',
            '--experiment_name', 'debug_',
            '--verbose',
            '--delta_check',
            '--batch_size', '64',
            '--lr_tr', '0.001'
        ]
    )
    """
    # CLARO DEBUGGER
    # '--data_dir', '/mimer/NOBACKUP/groups/snic2022-5-277/ltronchin/data',
    # '--box_apply',

def running():
    print("".center(100, '*'))
    print(" RUNNING CODE ".center(100, '*'))
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






    #  _______________________________________________________________________________________________
    # System Settings
    # Put here debugging parametrization
    torch.backends.cudnn.benchmark = True
    util_general.print_CUDA_info()
    sys.path.extend(["./"])
    # Seed everything
    util_general.seed_all()
    #  _______________________________________________________________________________________________
    # Experiment Options
    OptionstTrain = TrainOptions()
    opt = OptionstTrain.parse()

    # Change the working directory
    os.chdir(opt.workdir)

    #  _______________________________________________________________________________________________
    # Submit run:
    print("Submit run")
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d")

    # ------------------------------------------------------------------------------------------------
    log_path = os.path.join(opt.path_man.get_main_path(), 'log_run')
    run_id = get_next_run_id_local(os.path.join(log_path, opt.dataset_name), opt.phase)  # GET run id
    run_name = "{0:05d}_{1}_EXP_{2}_{3}".format(run_id, opt.phase, opt.id_exp, date_time)
    log_dir_exp = os.path.join(log_path, run_name)
    util_general.mkdir(log_dir_exp)
    # Initialize Logger - run folder
    OptionstTrain.print_options(opt=opt, path_log_run=log_dir_exp)
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
    #  _______________________________________________________________________________________________
    # Dataset Options
    dataset = create_dataset(opt)
    opt.dataset_size = dataset.__len__()
    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    #           ITERATIVE TRAINING / PRETRAINING
    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    dict_phase = {"train": 1, "pretrain": 0}
    if dict_phase[opt.phase]:
        print(opt.verbose)
        iterative_training_over_k()
    else:
        # Model Definition
        model = create_model(opt=opt)
        model.setup(opt=opt)
        pretrain()

