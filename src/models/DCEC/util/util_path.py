""" Class to manage the Folders/Subfolders Paths for wxperiments"""
import ntpath
import os
import shutil
import pathlib

from easydict import EasyDict
from util import util_general


class PathManager(object):
    """ WELCOME TO PathMAN
               ,##.                   ,==.
         ,# PATH #.               \ o ',
        #   PATH  #     _     _    \    \
        #   PATH  #    (_)   (_)   /    ;
         `#    #'                 /   .'
           `##'                   "=="
    This class implements a series of fuctions aided to create the architecture of folders needed to
    realize the experimental setup
    Parameters:
        opt (Option class)-- Options needed to complete the setup of the dataset.
    """

    def __init__(self, opt):
        # Options
        self.opt = opt
        # ---------------- Main directory of the experiment
        self.main_dir = os.path.join(opt.reports_dir, opt.name + '_' + opt.dataset_name)
        # Fix the main experimental folder, if it's the default one, experiments will be saved inside <./src/reports> folder.
        self.paths = EasyDict()
        # This state defines which sub folder of the main wxperiment has to be taken. It could be pretrain/train/
        # set the phase reference in the paths dictionary tree.
        self.set_phase(phase=self.opt.phase)
        self.exp_name_assignment()

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
                util_general.mkdir(phase_dir)  # phase directory creation
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
            util_general.mkdir(phase_dir)
            self.dict_phase.__setattr__("%s_dir" % "phase", phase_dir)
            print("The phase %s has been created in the paths dictionary!" % (phase))
            return

    def initialize_test_folders(self):
        directories = ['tables', 'plots']
        list(map(util_general.mkdir, [os.path.join(self.get_path('save_dir'), dir) for dir in directories]))
        for dir, path in zip(sorted(directories), filter(os.path.isdir, os.scandir(self.get_path('save_dir')))):
            self.set_dir(dir_to_extend="%s" % 'save_dir', path_ext=dir)
        return self

    def initialize_model(self, model=str()):
        """ Initialize sub-foldings
            Parameters:
                model (str): manually inserted model name
        """
        # directories tree division.

        directories = ['weights', 'plots', 'logs']  # create 3 subfoldings for weights files, logs  and plots.
        model_dir = "%s_dir" % (model if model.__len__() > 0 else "model")

        self.dict_phase.__setattr__(model_dir, os.path.join(self.get_path('save_dir'), self.opt.AE_type if model.__len__() == 0 else model))  # Model directory
        list(map(util_general.mkdir, [os.path.join(self.get_path(model_dir), dir) for dir in directories]))  # Create three subfolders for weights, plots and logs.
        # Extend directories tree from model directory.
        for dir, path in zip(sorted(directories), filter(os.path.isdir, os.scandir(self.get_path(model_dir)))):
            self.set_dir(dir_to_extend="%s" % model_dir, path_ext=dir)
        return self

    @staticmethod
    def clean_dir(path):
        """Clean directory from all files and all subdirs
            Parameters:
                  path (str): path string to delete."""
        shutil.rmtree(path)

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
            util_general.mkdir(new_path)  # create directory if not exist
            if os.path.exists(new_path) and force:
                self.clean_dir(new_path)
            self.dict_phase.__setattr__("%s_dir" % (name_att if name_att != "" else path_ext), new_path)

            return print("The %s folder successfully created extending the path: %s ." % (path_ext, path))

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

    def exp_name_assignment(self):

        """ Function that auto enumerate experiment by existing folders on disk
        This function generate tree of path based omn an auto_enumeration sistem:
            - Exp-ID subpath
            - save_dir subpath

        """
        phase_dir = self.get_path('phase_dir')
        list_paths = os.listdir(phase_dir)
        IDs_ = [EXP_directory.split('_')[1] for EXP_directory in list_paths]
        self.set_dir(dir_to_extend='phase_dir', name_att="save", path_ext=f"EXP_{self.opt.id_exp}")
        print(IDs_)



    def __repr__(self):
        return self.__class__.__name__


PIPE = "|"
ELBOW = "|__"
TEE = "|--"
PIPE_PREFIX =  "|   "
SPACE_PREFIX = "    "


# Directory Tree Printer / Organizer
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class DirectoryTree:
    def __init__(self, root_dir):
        self._generator = _TreeGenerator(root_dir)

    def generate(self):
        tree = self._generator.build_tree()
        for entry in tree:
            print(entry.encode('utf-8'))
        return tree
class _TreeGenerator:
    """
    <_TreeGenerator> Object Class is needed to generete the tree structure of the directory structure.
    FOr the pourpose of the tre construction there are some fixed attributeds
    Project added from https://realpython.com/directory-tree-generator-python/
    """

    def __init__(self, root_dir):
        """
        Class <__init__> constructor. In this case, .__init__() takes root_dir as an argument. It holds the tree’s root directory path.
         Note that you turn root_dir into a pathlib.Path object and assign it to the nonpublic instance attribute ._root_dir.
        :param root_dir: <pathlib.Path> Object
        """
        self._root_dir = pathlib.Path(root_dir)
        self._tree = [] # empty tree of directories

    def build_tree(self):
        """
        Tree building function.
        This public method generates and returns the directory tree diagram. Inside .build_tree(), you first call ._tree_head() to build the tree head.
        Then you call ._tree_body() with ._root_dir as an argument to generate the rest of the diagram.
        :return:
        list, the tree of directories
        """
        self._tree_head()
        self._tree_body(self._root_dir)
        return self._tree

    def _tree_head(self):
        """
        This method adds the name of the root directory to ._tree. Then you add a PIPE to connect the root directory to the rest of the tree.
        :return: None
        """
        self._tree.append(f"{self._root_dir}{os.sep}")
        self._tree.append(PIPE)

    def _tree_body(self, directory, prefix=""):
        """

        :param directory:  holds the path to the directory you want to walk through. Note that directory should be a pathlib.Path object.
        :param prefix: holds a prefix string that you use to draw the tree diagram on the terminal window.
        This string helps to show up the position of the directory or file in the file system.
        :return: None
        """
        entries = directory.iterdir()
        entries = sorted(entries, key=lambda entry: entry.is_file()) # This function iterate the generator of paths <entries>
        entries_count = len(entries)
        for index, entry in enumerate(entries):
            connector = ELBOW if index == entries_count - 1 else TEE
            if entry.is_dir():
                self._add_directory(
                    entry, index, entries_count, prefix, connector
                )
            else:
                self._add_file(entry, prefix, connector)

    def _add_directory(
            self, directory, index, entries_count, prefix, connector
    ):
        self._tree.append(f"{prefix}{connector} {directory.name}{os.sep}")
        if index != entries_count - 1:
            prefix += PIPE_PREFIX
        else:
            prefix += SPACE_PREFIX
        self._tree_body(
            directory=directory,
            prefix=prefix,
        )
        self._tree.append(prefix.rstrip())

    def _add_file(self, file, prefix, connector):
        self._tree.append(f"{prefix}{connector} {file.name}")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def find_exp(path_log_run:str, MODE):
    if MODE == 'last':
        import re
        # last path is equal to the path_log
        for sub_folder in os.listdir(path_log_run):
            res = int(re.search(r'\d+', sub_folder).group())
            if not isinstance(res, int): return path_log_run
        last_max = int(max(os.listdir(path_log_run)))
        last_path = os.path.join(path_log_run, str(last_max))
        find_exp(path_log_run=last_path, MODE=MODE)




def get_next_run_id_local(run_dir_root: str, module_name: str) -> int:
    """Reads all directory names in a given directory (non-recursive) and returns the next (increasing) run id. Assumes IDs are numbers at the start of the directory names."""
    import re
    #dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d))]
    #dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d)) and d.split('--')[1] == module_name]
    dir_names = []
    util_general.mkdir(run_dir_root)
    for d in os.listdir(run_dir_root):
        if not 'configuration.yaml' in d and not 'log.txt' in d and not 'src' in d:
            try:
                if os.path.isdir(os.path.join(run_dir_root, d)) and d.split('--')[1] == module_name:
                    dir_names.append(d)
            except IndexError:
                if os.path.isdir(os.path.join(run_dir_root, d)):
                    dir_names.append(d)

    r = re.compile("^\\d+")  # match one or more digits at the start of the string
    run_id = 1

    for dir_name in dir_names:
        m = r.match(dir_name)

        if m is not None:
            i = int(m.group())
            run_id = max(run_id, i + 1)

    return run_id
def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]
def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
def split_dos_path_into_components(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)

            break

    folders.reverse()
    return folders