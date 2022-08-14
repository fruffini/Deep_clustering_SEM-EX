""" Mix general pourpose functions"""
import os
import shutil
import random
import numpy as np
import torch
import yaml

# Function to load yaml configuration file

def load_config(config_file):
    """
    Loading config YAML file from "./configs" folder
    :param config_file: path (str) -- single path file
    :return: config_file (dict)
    """

    CONFIG_PATH = "./configs"
    with open(os.path.join(CONFIG_PATH, config_file)) as file:
        config = yaml.safe_load(file)
    return config


def mkdirs(paths: list):
    """create empty paths if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path: str):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def del_dir(path):
    """delete all the folders after the defined path

       Parameters:
           path (str) -- a single directory path
       """
    if os.path.exists(path):
        shutil.rmtree(path)

def seed_all(seed=None):  # for deterministic behaviour
    if seed is None:
        seed = 42
    print("Using Seed : ", seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.empty_cache()
    torch.manual_seed(seed)   # Set torch pseudo-random generator at a fixed value
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)   # Set numpy pseudo-random generator at a fixed value
    random.seed(seed)   # Set python built-in pseudo-random generator at a fixed value
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
