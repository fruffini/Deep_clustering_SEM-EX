# Implementation of DEEP-CONVOLUTIONAL-EMBEDDED_CLUSTERING for semantic-extraction procedure
# defined in the method SEM-EX
# hello here

from __future__ import print_function, division
import sys
import time
from tqdm import tqdm
from options.train_options import TrainOptions
from models import create_model
from src.models.DCEC.data import create_dataset
from torchvision import transforms
from src.utils.util_general import *


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
    exit_ = False # training exit condition
    total_iters = 0  # the total number of training iterations
    batch_iters = 0  # total batch processed
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs
        # Train the model for each value inside the k_values option list
        epoch_iter = 0
        model.update_learning_rate() if not epoch_iter == 0 else model.do_nothing()  # update learning rates linked to optimizer,
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
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
                    model.save_image_reconstructed(epoch=epoch)

            model.print_current_losses(epoch=epoch, iters=epoch_iter)
            model.reset_accumulator()
            if exit_:
                break
            else:
                continue
if __name__ == '__main__':
    print(sys.argv)
    sys.path.extend(["./"])
    # Seed everything

    print(os.getcwd())
    # Debugging Only.
    sys.argv.extend([
        '--phase','pretrain',
        '--AE_type', 'CAE3',
        '--dataset_name',
        'MNIST'])
    #

    opt = TrainOptions().parse()
    opt.img_shape = (28, 28)
    opt.verbose = False

    model = create_model(opt=opt)
    model.setup(opt=opt)


    dataset = create_dataset(opt)
    dataset.set_tranform(transform=transforms.ToTensor())
    opt.dataset_size = dataset.__len__()
    model.set_pretrain_folders()
    if opt.phase == "train":
        if model.load_model_pretrained():
            train()
        else:
            pretrain()
            train()
    elif opt.phase == "pretrain":
        pretrain()
    # pretrain procedure


    # ------ ITERATIVE TRAINING OVER k ------
    # Training for k_num of clusters variable through list values of parameters' list.
    # k parameters values
    opt.k_values = list(np.arange(2, opt.k_max + 1))
    for i, k in enumerate(opt.k_values):    # outer loop for different model instanced with different cluster number intialization MODEL_k -> MODEL_k+1

        print("HELLO")




















