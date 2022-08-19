# Implementation of DEEP-CONVOLUTIONAL-EMBEDDED_CLUSTERING for semantic-extraction procedure
# defined in the method SEM-EX
# hello here
from __future__ import print_function, division
import sys
import time
import os
from tqdm import tqdm

from util.util_path import get_next_run_id_local
from options.train_options import TrainOptions
from models import create_model
from util import util_general
from dataset import create_dataset
from torchvision import transforms

import numpy as np

def pretrain():
    # ----- PRETRAIN ------
    total_iters = 0  # the total number of training iterations
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
    model.prepare_training(dataloader=dataset.dataloader)
    exit_ = False  # training exit condition
    total_iters = 0  # the total number of training iterations
    batch_iters = 0  # the total number of batch processed
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs
        # Train the model for each value inside the k_values option list
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0  # the total number of training iterations during a single epoch
        model.update_learning_rate() if not total_iters == 0 else model.do_nothing()  # update learning rates linked to optimizers.
        with tqdm(dataset.dataloader, unit="batch", desc="Progress bar training phase") as tqdm_train:
            for ind, data in enumerate(tqdm_train):
                tqdm_train.set_description(f"Epoch {epoch}")
                # inner loop within one epoch
                if batch_iters % opt.update_interval == 0:
                    delta_bool = model.update_target()
                    if total_iters > 0 and delta_bool:
                        print('\nReached tolerance threshold. Stopping training.\n', flush=False)
                        exit_ = True
                        break
                # TODO print_freq, latest_freq sono utili ? Posso rimuovere ?
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
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
                    model.save_image_reconstructed(epoch=epoch)

            model.print_current_losses(epoch=epoch, iters=epoch_iter)
            model.reset_accumulator()
            if exit_:
                break
            else:
                continue


def iterative_training_over_k():
    # ------ ITERATIVE TRAINING OVER k ------
    # Training for k_num of clusters variable through list values of parameters' list.
    # k parameters values
    global model
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

        if opt.phase == "train":  # train phase
            if model.load_model_pretrained():
                train()
                print(
                    f"\n _______________________________________________________________________________________________ "
                    f"\n INFORMATION: the DCEC model with the number of clusters equal to {k} has been trained.  "
                    f"\n _______________________________________________________________________________________________"
                    )
            else:
                raise NotImplementedError(" Pretrained weights not implemented, please launch experiment with --phase <pretrain>")

        #  _______________________________________________________________________________________________
        #  _______________________________________________________________________________________________



if __name__ == '__main__':
    #  _______________________________________________________________________________________________
    # System Settings
    sys.path.extend(["./"])

    # Seed everything
    util_general.seed_all()
    print("Running path for the experiment:", os.getcwd())
    # Debugging Only.
    """sys.argv.extend(
        [
            '--phase', 'train',
            '--AE_type', 'CAE3',
            '--dataset_name',
            'CLARO',
            '--reports_dir', 'C:\\Users\\Ruffi\\Desktop\\Deep_clustering_SEM-EX\\reports',
            '--config_dir', 'C:\\Users\\Ruffi\\Desktop\\Deep_clustering_SEM-EX\\configs',
            '--data_dir', 'C:\\Users\\Ruffi\\Desktop\\Deep_clustering_SEM-EX\\data'
            ]

    )"""
    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    # Experiment Options
    OptionstTrain = TrainOptions()
    opt = OptionstTrain.parse()
    opt.img_shape = (28, 28)
    opt.verbose = False
    opt.n_epochs = 3  # Debug
    opt.n_epochs_decay = 3  # Debug
    #  _______________________________________________________________________________________________
    # Submit run:
    print("Submit run")
    run_module = os.path.basename(__file__)
    run_id = get_next_run_id_local(os.path.join('log_run', opt.dataset_name), opt.phase)  # GET run id
    run_name = "{0:05d}--{1}--EXP_{2}".format(run_id, run_module, opt.id_exp)
    log_dir = os.path.join('log_run', opt.dataset_name, run_name)
    util_general.mkdir(log_dir)
    #  _______________________________________________________________________________________________
    # Initialize Logger - run folder
    OptionstTrain.print_options(opt=opt, path_log_run=log_dir)
    logger = util_general.Logger(file_name=os.path.join(log_dir, 'log.txt'), file_mode="w", should_flush=True)
    #  _______________________________________________________________________________________________
    # Welcome
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    print("Hello!", date_time)

    # Dataset Options
    dataset = create_dataset(opt)
    #dataset.set_tranform(transform=transforms.ToTensor())
    opt.dataset_size = dataset.__len__()

    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    #           ITERATIVE TRAINING / PRETRAINING
    #  _______________________________________________________________________________________________
    #  _______________________________________________________________________________________________
    dict_phase = {"train": 1, "pretrain": 0}
    if dict_phase[opt.phase]:
        iterative_training_over_k()
    else:
        # Model Definition
        model = create_model(opt=opt)
        model.setup(opt=opt)
        pretrain()

