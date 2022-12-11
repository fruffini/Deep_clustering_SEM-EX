import os
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch
from easydict import EasyDict
from models import networks


class BaseModel(ABC):
    """This class is an abstract class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>: initialize the class calling BaseModel.__init__(self, opt).
        -- <set_input>: unpack data from dataset and aplly preprocessing
        -- <forward>: produce intermediate results.
        -- <optimize parameters>: calculate losses, gradients and update network weights.

    """
    def __init__(self, opt):
        """Initializae the BaseModel class.
        Parameters:
             opt (Option class)-- stores all the esperiments flags; need to be a subclass of Baseoptions.

        """

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.reports_dir, opt.experiment_name)  # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = EasyDict()
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def encode(self):
        """Run encoding pass; called by both functions <optimize_parameters> and <test>."""
        pass


    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass
    @abstractmethod
    def accumulate_losses(self):
        """Accumulate losses"""
        pass
    @abstractmethod
    def compute_metrics(self):
        """Compute metrics"""
        pass


    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def get_path_phase(self, name, phase=None):
        """Function to generate tree off folders to save experiments"""

        return self.opt.path_man.get_path(name=name) if phase is None else self.opt.path_man.get_path_phase(name=name, phase=phase)

    def get_current_losses(self):
        """Return training losses/errors."""
        losses = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses[name] = float(getattr(self, name + '_loss'))
        return losses

    def do_nothing(self):
        """Do nothing method"""
        pass

    def set_losses_dict(self, losses):
        """Set the dictionary to store the history of losses during the training"""
        self.losses_dict = {loss: list for loss in self.loss_names}
    def set_metrics_dict(self):
        """"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
        # Add schedulers to class BaseModel.
            self.schedulers = {key_phase: networks.get_scheduler(optimizer, opt) for key_phase, optimizer in self.optimizers.items()}
        self.print_networks(opt.verbose)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        phase = self.opt.phase
        old_lr = self.optimizers[phase].param_groups[0]['lr']
        if self.opt.lr_policy == 'plateau':
            self.schedulers[phase].step(self.metric)
        else:
            self.schedulers[phase].step()
        lr = self.optimizers[phase].param_groups[0]['lr']
        if self.opt.verbose:
            print('optimizer: %.7s  --learning rate %.7f -> %.7f' % (self.optimizers[phase].__class__.__name__, old_lr, lr) if not old_lr==lr else 'Learning rate non modificato: %s' %(old_lr))

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.get_path_phase("weights_dir"), save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch, path_ext=str):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        Returns:
            bool -- indicates the presence or not of trained networks.
        """
        returned = False
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir if path_ext is None else path_ext, load_filename)

                if not os.path.exists(load_path):
                    print("Pesi non presenti : %s" %load_filename)
                    continue
                else:
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                    net.load_state_dict(state_dict)  # loading model weights
                    returned = True
        return returned

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)

                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths