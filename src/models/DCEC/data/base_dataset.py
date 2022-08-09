"""This module implementes an abstract class ABC 'BaseDataset for datasets"""
import random
from abc import ABC, abstractmethod

import torch.utils.data as data
from numba.core.types import abstract


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create subclass, you need to implement the following 4 functions:
    -- <__init__>:                          initialize the class, first call BaseDataset.__init__(self,opt)
    -- <__len__>:                           return the size of the dataset.
    -- <__getitem__>:                       get a data point.
    -- <modify_commandline_options>:        (optionally) add dataset specific options ad set default options.

    """
    def __init__(self, opt):
        """Constructor call function to initialize the class
        Parameters:
            opt (Option class)-- stores all the experiments flags; subclass of BaseOptions
        """
        self.opt = opt

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific
            or test-specific options.

        Returns:
            the modified parser.
        """
        return parser
    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

