""" Class to manage the Folders/Subfolders Paths for wxperiments"""
import os
import shutil

from easydict import EasyDict
from src.utils.util_general import mkdirs, mkdir, del_dir


class PathManager(object):
    """ WELCOME TO PathMAN
               ,##.                   ,==.
         ,# PATH #.               \ o ',
        #   PATH  #     _     _    \    \
        #   PATH  #    (_)   (_)   /    ;
         `#    #'                 /   .'
           `##'                   "=="

    """

    def __init__(self, opt):

        # Options
        self.opt = opt
        # Main directory of the experiment
        self.main_dir = os.path.join(opt.reports_dir, opt.name + '_' + opt.dataset_name)
        self.paths = EasyDict()
        # This state defines which sub folder of the main wxperiment has to be taken. It could be pretrain/train/
        # set the phase reference in the paths dictionary tree.
        self.set_phase(phase=self.opt.phase)
        self.auto_enumerate()

    def set_phase(self, phase):
        """
        Settings the phase in the path manager
        :param phase (str): the new phase string to set in the path Manager.
        """
        self.state = phase
        self.check_phase_paths()  # check if paths have just been set inside the path manager
        self.dict_phase = self.get_dict_phase(phase=self.state)  # the dictionary phase is a pointer to the phase dictionary of paths.

    def get_dict_phase(self, phase):
        """Return the dictionary in relation with the specific phase
            Parameters:
                phase (str): defines the phase. pretrain/train/test
            :returns
            <class EasyDict>, paths inside the phase folder of the dictionary
        """
        return self.paths.__getattribute__(phase)

    def change_phase(self, state):
        states = ['pretrain', 'train', 'test']
        previous_state = self.state
        assert (state in states)
        self.state = state
        self.set_phase(phase=state)
        return print("State changed from : %s , to : %s" % (previous_state, state))

    def check_phase_paths(self):
        """Intialize the phase sub-directories"""
        phase = self.state
        if self.paths.__len__() > 0:
            if not phase in self.paths.keys():
                # Different from first initialization of the phase
                self.paths['{}'.format(phase)] = EasyDict()
                self.dict_phase = self.get_dict_phase(phase=self.state)
                phase_dir = os.path.join(self.main_dir, phase)
                mkdir(phase_dir)  # phase directory creation
                self.dict_phase.__setattr__("%s_dir" % "phase", phase_dir)
                print("The phase %s has been created in the paths dictionary!" % (phase))
                return
            else:
                print("Phase paths already exists!")
                return

        else:
            # First initialization of the phase
            self.paths['{}'.format(phase)] = EasyDict()
            self.dict_phase = self.get_dict_phase(phase=self.state)
            # phase directory creation and insertion into path manager
            phase_dir = os.path.join(self.main_dir, phase)
            mkdir(phase_dir)
            self.dict_phase.__setattr__("%s_dir" % "phase", phase_dir)
            print("The phase %s has been created in the paths dictionary!" % (phase))
            return



    def initialize_model(self, model=str()):
        """ Initialize sub-foldings
            Parameters:
                model (str): manually inserted model name
        """
        # directories tree division.
        directories = ['weights', 'plots', 'logs']  # create 3 subfoldings for weights files, logs  and plots.
        model_dir = "%s_dir" % (model if model.__len__() > 0 else "model")
        self.dict_phase.__setattr__(model_dir, os.path.join(self.get_path('save_dir'), self.opt.AE_type if model.__len__() == 0 else model))  # Model directory
        list(map(mkdir, [os.path.join(self.get_path(model_dir), dir) for dir in directories]))  # Create three subfolders for weights, plots and logs.
        # Extend directories tree from model directory.
        for dir, path in zip(sorted(directories), filter(os.path.isdir, os.scandir(self.get_path(model_dir)))):
            self.set_dir(dir_to_extend="%s" % model_dir, path_ext=dir)
        return self

    @staticmethod
    def clean_dir(path):
        """Clean directory from all files and all subdirs
            Parameters:
                  path (str): path string to delete."""
        with os.scandir(path) as scan:
            list(os.remove(pe) for pe in scan)

    def set_dir(self, dir_to_extend, path_ext, name_att="", force=False):
        """Set new attribute in <self.paths> by string and create a new folder.
            Parameters:
                  path_ext (str): name of the new extension.
                  path_to_extend (str): path to extend
                  force (boolean) : force the clean up of the folder

        """
        path = self.get_path(dir_to_extend)
        if isinstance(path_ext, str):
            new_path = os.path.join(path, path_ext)
            mkdir(new_path)  # create directory if not exist
            if os.path.exists(new_path) and force:
                self.clean_dir(new_path)
            self.dict_phase.__setattr__("%s_dir" % (name_att if name_att != "" else path_ext), new_path)

            return print("The %s folder successfully created extending the path: %s ." % (path_ext, dir_to_extend))

    def get_path_phase(self, name, phase):
        """ Get method that returns the items inside the dictionary of paths in relation to the selected phase
            Parameters:
                name (str): name of the path attribute in the dictionary

        """
        return self.paths.__getattribute__(phase).__getattribute__(name)

    def get_path(self, name):
        """ Get method tha returns the items inside the dictionary of paths
            Parameters:
                name (str): name of the path attribute in the dictionary

        """
        return self.dict_phase.__getattribute__(name)

    def auto_enumerate(self):

        """ Function that auto enumerate experiment by existing folders on disk """
        phase_dir = self.get_path('phase_dir')
        list_paths = os.listdir(phase_dir)
        self.ID_max = None
        if not list_paths.__len__() == 0:
            for EXP_directory in list_paths:
                if self.opt.id_exp in EXP_directory.split('_')[1]:
                    self.set_dir(dir_to_extend='phase_dir', name_att="save", path_ext=EXP_directory)
                    return
                elif self.opt.id_exp == 'auto':
                    # In auto mode the manager build a new root where to save the experiment, and assign the ID_# number searching for maximum ones.
                    self.ID_max = max([int(''.join(filter(str.isdigit, path))) for path in list_paths]) if self.ID_max is not None else self.ID_max
                    self.set_dir(dir_to_extend='phase_dir', name_att="save", path_ext="EXP_ID{}".format(self.ID_max + 1))
                    return
                else:
                    raise NotImplementedError('{} not implemented'.format(self.opt.id_exp))
        else:
            self.set_dir(dir_to_extend='phase_dir', name_att="save", path_ext="EXP_ID{}".format(1))
            return

    def __repr__(self):
        return self.__class__.__name__
