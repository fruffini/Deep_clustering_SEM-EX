import argparse
from util.util_general import *
import models, dataset


# Translate string entries to bool for parser
from src.models.DCEC.util.util_path_manager import PathManager


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseOptions(object):
    """This class defines options used during both training and test time. It defines the general porpouse options
    for the script launch/testing.



    """
    def __init__(self, path_manager=None):
        """Reset the class: indicates tha the calss has not been initialized.
        """
        self.initialized = False


    def initialize(self, parser):
        """ Define the common options for both training/testing script"""

        # basic parameters
        #parser.add_argument('--dataroot', required=True,
        #    help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name',
            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--use_wandb', action='store_true', help='use wandb')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # Directories outside the src module
        parser.add_argument('--reports_dir', type=str, default='./reports', help='Customized report direcrtory folder, else it would be put like /src/reports ' )
        parser.add_argument('--config_dir', type=str, required=True, default="C:\\Users\\Ruffi\\Desktop\\Deep_clustering_SEM-EX\\configs", help='configs files folder IMPORTANT:'
                                                                                                                                                ' 1) Dataset config.yaml, '
                                                                                                                                                '2) Autoencoders_layers_config.yaml. ')
        # model parameters
        parser.add_argument('--model', default='DCEC', choices=['DCEC', 'VAE'], help='Model name for the experiment.')
        parser.add_argument('--AE_cfg_file', default='AE_layers_settings.yaml',
             help='Layers setting file for CAE')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--embedded_dimension', type=int, default=10, help='Dimension of bottleneck latent space AE.')
        parser.add_argument('--alpha', type=float, default=1.0,
            help='Alpha coefficient for t-Student probability distribution.')
        parser.add_argument('--rec_type', type=str, default='bce', choices=['mse', 'bce', 'vanilla'], help='Type of '
                                                                                                         'reconstruction loss function.')
        parser.add_argument('--AE_type', type=str, default='CAE2', choices=['CAE2', 'CAE3', 'CAE4'],
            help='type of architecture for the Autoencoder')
        parser.add_argument('--activations', type=str, default='sigmoid', choices=['sigmoid', 'tanh', 'none'],
            help='Activations for Autoencoder.')
        parser.add_argument('--norm', type=str, default='none',help='q_ij normalization or none [ '
                                                                 'q_ij | none]')
        parser.add_argument('--init_type', type=str, default='normal',
            help='network initialization [normal | xavier | kaiming]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for net\'s init function')
        parser.add_argument('--dropout', type=int, default=0.0,  help='no dropout for the Autoencoder')
        parser.add_argument('--leaky', action='store_true', help='no dropout for the Autoencoder')
        parser.add_argument('--leaky_nslope', type=int, default=0.15, help='Negative sloper for Leaky '
                                                                                   'Relu activation layer')

        # ADDITIONAL PARAMETERS
        # Type of learning phase

        parser.add_argument('--phase', type=str, default='pretrain',choices=['pretrain', 'train', 'test'], help='Choosing between Pretrain/Train/Test phase of model.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--epoch', type=str, default='latest',
            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0',
            help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; '
                 'otherwise, the code will load models by [epoch]')
        parser.add_argument('--suffix', default='', type=str,
            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        # Dataset images
        parser.add_argument('--id_exp', type=str, default='ID1', choices=['auto', 'ID#'])
        parser.add_argument('--dataset_name', required=True, default='MNIST', choices=['MNIST', 'CLARO'], help='CLARO or MNIST')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=32, help='input q_ij size')
        parser.add_argument(
            '--max_dataset_size', type=int, default=float("inf"),
            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.'
            )
        self.initialized = True
        return parser
    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults
        # modify dataset-related parser options
        dataset_name = opt.dataset_name
        dataset_option_setter = dataset.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain # train or test
        # Initialize the Paths Manager
        self.path_manager = PathManager(opt=opt)
        opt.path_man =  self.path_manager
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix


        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """


        message = ''
        message += f'----------------- Options: {opt.phase} ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        file_name = os.path.join(opt.path_man.get_path(opt.model_dir), '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


