from __future__ import print_function
from src.models.DCEC.data.base_dataset import BaseDataset
from PIL import Image
import os.path
import errno
import codecs
from src.utils.util_general import *
from easydict import EasyDict as edict

class MNISTDataset(BaseDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

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
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, opt, transform=None, target_transform=None):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.config = edict(load_config('MNIST_configuration.yaml'))
        self.root = os.path.expanduser(self.config.data.root)
        self.type = opt.mnist_mode
        self.transform = transform
        self.target_transform = target_transform
        self.train = opt.isTrain  # training set or test set
        self.full = False
        if not self._check_exists():
            print('Dataset not found.')
            self.download()
        else:
            print('Dataset found in location %s' %self.root)
        # Train/Test data images and labels.
        self.train_data, self.train_labels = torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
        self.test_data, self.test_labels = torch.load(os.path.join(self.root, self.processed_folder, self.test_file))

        if self.type == ('full' or 'small'):
            self.train_data = np.concatenate((self.train_data, self.test_data), axis=0)
            self.train_labels = np.concatenate((self.train_labels, self.test_labels), axis=0)
            self.full = True

        if self.type == 'small':
            index = np.floor(opt.perc * (len(self.train_data) + len(self.test_data)))
            self.train_data = self.train_data[0:int(np.floor(index*0.8))]
            self.train_labels = self.train_labels[0:int(np.floor(index*0.8))]
            self.test_data = self.test_data[0:int(np.floor(index*0.2))]
            self.test_labels = self.test_labels[0:int(np.floor(index*0.2))]
            self.full = False
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
    def set_transform(self, transform=None):
        """ Adding transform function to dataset
        Parameters:
            transform (torchvision.transform): transform function
            TODO insert the possibility to add multi transformation for data augmentation.
        """
        self.transform = transform
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.full:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

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



def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
