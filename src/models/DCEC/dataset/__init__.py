import importlib
import random

import pandas as pd

from util.util_general import shuffle_and_index
import copy
import easydict
from torch.utils.data import Subset
import numpy as np
import torch
import os
from .base_dataset import BaseDataset
def find_dataset_using_name(dataset_name, target_suffix=None):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    if 'GLOBES' in dataset_name:
        dataset_load = 'GLOBES'
    else:
        dataset_load = dataset_name
    dataset_filename = "dataset." + dataset_load + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_load.replace('_', '') + 'dataset' if target_suffix is None else dataset_load + target_suffix
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset
def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create dataset given options in input
    This function wraps the class custom data loader
    """
    data_loader = CustomDatasetDataLoader(opt=opt)
    return data_loader.load_data()


def separate_dataset_by_label(opt, labels_info):
    """Create dataset given options in input
    This function wraps the class custom data loader
    """
    return DatasetLabelsDataLoader(opt=opt, labels_info=labels_info)


class DatasetLabelsDataLoader(object):
    def __init__(self, opt, labels_info: pd.DataFrame):
        self.opt = opt
        if 'GLOBES' in opt.dataset_name:
            dataset_load = 'GLOBES'
        else:
            dataset_load = opt.dataset_name
        target_class = 'labels'

        dataset_class = find_dataset_using_name(dataset_load, target_suffix=f'dataset{target_class}')
        self.datasets_by_label = dict()
        for k_ in np.arange(self.opt.num_clusters):
            print('INFO: Label : ', k_, 'Separated from dataset', dataset_load)
            label_k_dataset = labels_info[labels_info['clusters_labels'] == k_]
            self.datasets_by_label['Dataset_label_{}'.format(k_)] = dataset_class(opt, label_k_dataset)
    def get_labels_dict(self):
        return self.datasets_by_label


class CustomDatasetDataLoader():
    """Wrapper class of dataset class that perform multithreading data loading"""
    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt

        if 'GLOBES' in opt.dataset_name:
            dataset_load = 'GLOBES'
        else:
            dataset_load = opt.dataset_name

        print('INFO: Importing dataset name: ', dataset_load)
        dataset_class = find_dataset_using_name(dataset_load)
        self.dataset = dataset_class(opt)
        print("INFO: dataset [%s] was created" % type(self.dataset).__name__)
        self.split_train = 0.9

        # ADD a column to track the original order of the dataset
        try:
            assert self.opt.dataset_name != 'MNIST'
            self.df_data = self.dataset.data
            dict_mapper = {id: i for i, id in enumerate(self.df_data.index)}
            indexing = self.df_data.index.map(dict_mapper)
            self.df_data.insert(2, "index_number", list(indexing))
            self.dataset.data = self.df_data
            # indexing saved
            self.create_datasets_splitted()
        except:
            print('The Dataset MNIST is loading...')
        self.dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=opt.batch_size,
                shuffle=opt.shuffle_batches,
                num_workers=int(opt.num_threads))
        self.dataloader_big_batch = torch.utils.data.DataLoader(
                self.dataset, batch_size=256,
                shuffle=False,
                num_workers=int(self.opt.num_threads)
            )

    def get_new_indexig(self):
        if self.opt.dataset_name == 'MNIST':
            return np.array(self.index)
        print(" New Indexing:", self.dataset.data['index_number'])
        return np.array(self.dataset.data['index_number'])

    def shuffle_data(self):
        try:
            assert self.opt.dataset_name != "MNIST"
            self.dataset.data = self.dataset.data.sample(frac=1, random_state=int(os.environ['PYTHONHASHSEED']))
            self.df_data = self.dataset.data
        except:
            # FOR MNIST
            shuffled, index = shuffle_and_index(self.dataset.train_data)
            self.dataset.train_data = shuffled
            self.dataset.train_labels = np.array([np.array(self.dataset.train_labels[i]) for i in index])
            self.index = index
            pass


        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.num_threads))
        self.dataloader_big_batch = torch.utils.data.DataLoader(
            self.dataset, batch_size=256,
            shuffle=False,
            num_workers=int(self.opt.num_threads))
    def create_datasets_splitted(self):

        # Variables needed for the splitting
        original_dataset_len = len(self.dataset)
        n_tr_samples = original_dataset_len*self.split_train *100 // 100
        train_subset = Subset(self.dataset, np.arange(0, n_tr_samples))
        test_subset = Subset(self.dataset, np.arange(n_tr_samples, original_dataset_len))
        # Dataframe organizations
        DF_train = None
        DF_test = None
        if self.opt.dataset_name != 'MNIST':
            DF_train = self.dataset.data.iloc[train_subset.indices]
            DF_val = self.dataset.data.iloc[test_subset.indices]

        self.datasets_splitted = easydict.EasyDict(
            {'train':
                {
                    'dataset': train_subset,
                    'DataFrame': DF_train ,
                    'dataloader': torch.utils.data.DataLoader(train_subset,
                                                              batch_size=self.opt.batch_size,
                                                              shuffle=self.opt.shuffle_batches,
                                                              num_workers=int(self.opt.num_threads))
                },
             'val':
                {
                    'dataset': test_subset,
                    'DataFrame': DF_test,
                    'dataloader': torch.utils.data.DataLoader(train_subset,
                                                              batch_size=self.opt.batch_size,
                                                              shuffle=False,
                                                              num_workers=int(self.opt.num_threads))

                }
            }
        )





    def test_workers(self):
        from time import time
        import multiprocessing as mp

        print('\n' + '-'*5 + '\n' + '-'*3 + 'Testing workers:' + '-'*3 + '\n' + '-'*5)
        for num_workers in range(2, mp.cpu_count(), 2):
            loader_to_test = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.opt.batch_size,
            num_workers=num_workers, pin_memory=True, shuffle=True)
            start = time()
            for epoch in range(1, 3):
                for i, data in enumerate(loader_to_test, 0):
                    if i > 4:
                        break
                    pass
            end = time()
            print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
        del loader_to_test
    def load_data(self):
        """Loading data and returns the dataloader"""
        return self
    def concat_dataset(self):
        x_out=None
        y_out=None
        for data in self.dataloader:
            x_out = np.concatenate((x_out, data[0]), 0) if x_out is not None else data[0]
            y_out = np.concatenate((y_out, data[1]), 0) if y_out is not None else data[1]
        return x_out, y_out
    def set_tranform(self, transform):
        self.dataset.set_transform(transform=transform)
    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return q_ij of data"""
        for i, data in enumerate(self.dataloader):
            if i*self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data