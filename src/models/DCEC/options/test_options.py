from .base_options import BaseOptions

class TestOptions(BaseOptions):
    """This class includes training options:
    the option '--mode' defines the pretraining or training phase."""
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--threshold', type=int, default=95,
                            help='Probability threshold for the q_ij assignments by DCEC clustering module in order to define Clusters prototypes'
                            )
        self.isTrain = False
        return parser

    def print_options(self, opt, path_log_run):
        super().print_options(opt=opt, path_log_run=path_log_run)

