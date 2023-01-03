from util import util_path
from dataset import BaseDataset
from options.base_options import BaseOptions
from util.util_general import *
from easydict import EasyDict as edict
import torch
import numpy as np
from PIL import Image

import os
import random
import math
import pandas as pd


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
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
        if box[0] - border < 0:
            pad = 0 - (box[0] - border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2] + border > img_h:
            pad = (box[2] + border) - img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1] - diff_1 - border < 0:
            pad = 0 - (box[1] - diff_1 - border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3] + diff_2 + border > img_w:
            pad = (box[3] + diff_2 + border) - img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0] - border:box[2] + border, box[1] - diff_1 - border:box[3] + diff_2 + border]
    elif l_w > l_h:
        if box[0] - diff_1 - border < 0:
            pad = 0 - (box[0] - diff_1 - border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2] + diff_2 + border > img_h:
            pad = (box[2] + diff_2 + border) - img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1] - border < 0:
            pad = 0 - (box[1] - border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3] + border > img_w:
            pad = (box[3] + border) - img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0] - diff_1 - border:box[2] + diff_2 + border, box[1] - border:box[3] + border]
    else:
        if box[0] - border < 0:
            pad = 0 - (box[0] - border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2] + border > img_h:
            pad = (box[2] + border) - img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1] - border < 0:
            pad = 0 - (box[1] - border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3] + border > img_w:
            pad = (box[3] + border) - img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0] - border:box[2] + border, box[1] - border:box[3] + border]
    return img
"""def normalize(img, convert_to_uint8, scale_by_255, min_val=None, max_val=None):
    if not min_val:
        min_val = img.min()
    if not max_val:
        max_val = img.max()

    img = (img.astype(np.float64) - min_val) / (max_val - min_val)
    if scale_by_255:
        img = 255.0 * img
    if convert_to_uint8:
        img = img.astype(np.uint8)

    return img"""


def normalize(img, convert_to_uint8, scale_to_255, scale, **kwargs):
    if len(scale) != 0:
        min_val = scale['min']
        max_val = scale['max']
    else:
        min_val = img.min()
        max_val = img.max()

    img = (img.astype(np.float64) - min_val) / (max_val - min_val)
    if scale_to_255:
        img = 255.0 * img
    if convert_to_uint8:
        img = 255.0 * img
        img = img.astype(np.uint8)
    return img

def loader(**kwargs):

    k = edict(kwargs)
    img_path = k.img_path
    box = k.box
    img_dim = k.img_dim
    clip = k.clip
    # Img loading
    img = load_img(img_path)
    # Select Box Area
    if box is not None:
        img = get_box(img, box, perc_border=0.5)
    # Resize
    if img_dim != img.shape[0]:
        import cv2
        img = cv2.resize(img, (img_dim, img_dim))
    # Clip
    if clip is not None:
        img = np.clip(img, clip['min'], clip['max'])

    # Normalize
    img = normalize(img, **kwargs)

    # To Tensor
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    return img


class CLARODataset(BaseDataset):
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
            opt: BaseOptions,  # stores all the experiments flags; subclass of BaseOptions
            cip=dict,  # clip range
            scale=dict,  # sale range
            convert_to_uint8: bool = False,  # uint8 options      # scale from [0.0, 1.0] to [0.0 255.0]
            transform: object = None
    ):
        """Initialization"""
        BaseDataset.__init__(self, opt)
        # Config and Directories

        self.config = edict(load_config('CLARO_configuration.yaml', config_directory=self.opt.config_dir))
        opt.img_shape = (self.config['data']['image_size'], self.config['data']['image_size'])
        self.root_dir = self.opt.data_dir
        self.raw_dir = os.path.join(self.root_dir, self.config['data']['raw_dir'])
        self.interim_dir = os.path.join("/mimer/NOBACKUP/groups/snic2022-5-277/fruffini/data", self.config['data']['interim_dir'])
        self.transform = transform
        # Upload info claro
        self.info_claro = pd.read_excel(os.path.join(self.interim_dir, self.config['data']['data_info']))
        # Upload raw_data.
        self.data = pd.read_excel(os.path.join(self.interim_dir, self.config['data']['box_file']), index_col="img ID", dtype=list)  # Data Patient ID-slicesID

        if self.opt.box_apply:
            self.boxes = {os.path.basename(row[0]): eval(row[1][self.config['data']['box_value']]) for row in self.data.iterrows()}
        else:
            self.boxes = None

        # Box

        self.clip = self.config['data']['clip']
        self.convert_to_uint8 = self.config['data']['convert_to_uint8']
        self.scale_to_255 = self.config['data']['scale_to_255']
        self.scale = self.config['data']['scale']

        print("Dataset CLARO ready for train!")

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
        parser.add_argument(
            '--data_dir', type=str, default='C:\\Users\\Ruffi\\Desktop\\Deep_clustering_SEM-EX\\data\\claro', required=True, help='Input dataset directory, ./data default.'
            )
        parser.add_argument('--box_apply', action='store_true', help='if true, extract sub portion of the image, otherwise it does nothing')
        parser.add_argument('--image_shape', type=int, default=256, help='image dimension')
        parser.add_argument('--shuffle_interval', type=int, default=0, help='shuffle epochs interval')
        return parser

    def set_transform(self, transform=None):
        """ Adding transform function to dataset
        Parameters:
            transform (torchvision.transform): transform function
            TODO insert the possibility to add multi transformation for data augmentation.
        """
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        row = self.data.iloc[index]
        row_name = row.name.split('_')
        img_id = row_name[1]
        complete_id = row.name
        patient_id = row_name[0]
        # load box
        box = self.boxes[patient_id + '_' + img_id] if self.boxes else None
        # Load data and get label
        img_path = os.path.join(self.raw_dir, patient_id, "images", f"{patient_id + '_' + img_id}.tif")
        kwargs = {'img_path': img_path,
                  'img_dim': self.config['data']['image_size'],
                  'scale': self.scale,
                  'box': box,
                  'clip': self.clip,
                  'convert_to_uint8': self.convert_to_uint8,
                  'scale_to_255' : self.scale_to_255
                  }
        x = loader(
            **kwargs
            )

        if self.transform is not None:
            x = self.transform(x)
        return x, patient_id, complete_id


