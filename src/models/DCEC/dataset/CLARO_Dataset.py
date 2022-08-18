from options.base_options import BaseOptions
from util.util_general import *
from easydict import EasyDict as edict
import torch
import numpy as np
from PIL import Image
import cv2
import os
import random
import math
import pandas as pd


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_img(img_path):
    filename, extension = os.path.splitext(img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(float)
    return img


def get_box(img, box, perc_border=.0):
    # Sides
    l_h = box[2] - box[0]
    l_w = box[3] - box[1]
    # Border
    diff_1 = math.ceil((abs(l_h - l_w) / 2))
    diff_2 = math.floor((abs(l_h - l_w) / 2))
    border = int(perc_border * diff_1)
    # Img dims
    img_h = img.shape[0]
    img_w = img.shape[1]
    if l_h > l_w:
        if box[0]-border < 0:
            pad = 0-(box[0]-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+border > img_h:
            pad = (box[2]+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-diff_1-border < 0:
            pad = 0-(box[1]-diff_1-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+diff_2+border > img_w:
            pad = (box[3]+diff_2+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-border:box[2]+border, box[1]-diff_1-border:box[3]+diff_2+border]
    elif l_w > l_h:
        if box[0]-diff_1-border < 0:
            pad = 0-(box[0]-diff_1-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+diff_2+border > img_h:
            pad = (box[2]+diff_2+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-border < 0:
            pad = 0-(box[1]-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+border > img_w:
            pad = (box[3]+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-diff_1-border:box[2]+diff_2+border, box[1]-border:box[3]+border]
    else:
        if box[0]-border < 0:
            pad = 0-(box[0]-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+border > img_h:
            pad = (box[2]+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-border < 0:
            pad = 0-(box[1]-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+border > img_w:
            pad = (box[3]+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-border:box[2]+border, box[1]-border:box[3]+border]
    return img


def normalize(img, convert_to_uint8, scale_by_255, min_val=None, max_val=None):
    if not min_val:
        min_val = img.min()
    if not max_val:
        max_val = img.max()

    img = (img.astype(np.float64) - min_val) / (max_val - min_val)
    if scale_by_255:
        img = 255.0 * img
    if convert_to_uint8:
        img = img.astype(np.uint8)

    return img

def loader(img_path, img_dim, box, clip, scale, convert_to_uint8, scale_by_255):
    # Img
    img = load_img(img_path)

    # Select Box Area
    if box is not None:
        img = get_box(img, box, perc_border=0.5)
    # Resize
    if img_dim != img.shape[0]:
        img = cv2.resize(img, (img_dim, img_dim))
    # Clip
    if clip is not None:
        img = np.clip(img, clip['min'], clip['max'])

    # Normalize
    if scale is not None:
        img = normalize(img, convert_to_uint8, scale_by_255, min_val=scale['min'], max_val=scale['max'])
    else:
        img = normalize(img, convert_to_uint8, scale_by_255)

    # To Tensor
    img = torch.from_numpy(img)
    img  = img.unsqueeze(0)
    return img

class CLARODataset(torch.utils.data.Dataset):
    """`'CoLlAborative multi-sources Radiopathomics
    approach for personalized Oncology in non-small cell lung cancer' (CLARO)
      <http://www.cosbi-lab.it/claro/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(
            self,
            data:               pd.Series,      # patients id list as <id_patient>_<id_slice>
            opt:                BaseOptions,    # stores all the experiments flags; subclass of BaseOptions
            data_dir:           str,            # path to raw data
            resolution:         int,            # desidered resolution
            data_dir_box:       str,            # path to box file
            box_value:          str,            # box opt
            clip:               {},             # clip range
            scale:              {},             # sale range
            convert_to_uint8    :bool=False,    # uint8 options
            scale_by_255        :bool=True      # scale from [0.0, 1.0] to [0.0 255.0]
    ):
        """Initialization"""
        super().__init__(self, opt)
        self.config = self.config = edict(load_config('CLARO_configuration.yaml', config_directory=self.opt.config_dir))
        self.img_dir = os.path.join(data_dir)
        self.data = data # estrai da file patients_info_CLARO_retrospettivo.xlsx
        # Box
        if data_dir_box is not None:
            box_data = pd.read_excel(data_dir_box, index_col="img ID", dtype=list)
            self.boxes = {os.path.basename(row[0]): eval(row[1][box_value]) for row in box_data.iterrows()}
        else:
            self.boxes = None

        self.clip = clip
        self.scale = scale
        self.convert_to_uint8 = convert_to_uint8
        self.scale_by_255 = scale_by_255
        self.img_dim = resolution
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

                Parameters:
                    parser          -- original option parser
                    is_train (bool) -- whether training phase or test phase. You can use this flag to add
                    training-specific
                    or test-specific options.

                Returns:
                    the modified parser.
                """
        parser.add_argument('--mnist_mode', type=str, default='small', choices=['None', 'small', 'full'], help='Dataset loading options.')
        parser.add_argument('--perc', type=float, default=0.1, help='Percentage of dataset')
        return parser
    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        row = self.data.iloc[index].split('_')
        img_id = row[1]
        patient_id = row[0]

        # load box
        if self.boxes:
            box = self.boxes[patient_id + '_' + img_id]
        else:
            box = None
        # Load data and get label
        img_path = os.path.join(self.img_dir, patient_id, "images", f"{patient_id + '_' + img_id}.tif")
        x = loader(img_path=img_path, img_dim=self.img_dim, box=box, clip=self.clip, scale=self.scale, convert_to_uint8=self.convert_to_uint8, scale_by_255=self.scale_by_255)

        return x, patient_id, img_id
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
