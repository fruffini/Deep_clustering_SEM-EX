import itertools
from collections import OrderedDict
import pandas as pd
from PIL import Image
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from util import util_general
from util.util_clustering import kmeans, metrics_unsupervised_CVI
from .base_model import BaseModel
from .networks import *
import h5py
torch.autograd.set_detect_anomaly(True)


class DCECModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True, is_pretrain=False):
        """Add new model-specific options and rewrite default values for existing options.
            Parameters:
                parser -- the option parser
                is_train -- if it is training phase or test phase. You can use this flag to add training-specific
                or test-specific options.
                is_pretrain -- if it is pretraining phase only for AE or is training phase for DCEC.
            Returns:
                the modified parser.
            """
        parser.add_argument('--num_clusters', type=int, default=10, help='Num of clusters in which separate samples.')
        # Iterative learning
        parser.add_argument('--k_0', type=int, default=4, help='Starting number of centroids for the iterative training.')
        parser.add_argument('--k_fin', type=int, default=10, help='Final number of centroids for the iterative training.')
        parser.add_argument('--redo_encoding', action='store_true', help='Redo the encoding during the test phase')
        if is_train:

            # DCECs Parameters
            parser.add_argument('--update_interval', default=500, type=float, help='update iterations interval to update target distribution.')
            parser.add_argument('--gamma', default=0.3, type=float, help='clustering loss weight')
            parser.add_argument('--delta_label', default=0.0008, type=float, help='delta label stop condition between every update iteration interval.')
            parser.add_argument('--delta_check', action='store_true', help='if true, checks the delta label condition, otherwise it continue training until the last epoch')
            parser.add_argument('--delta_count', default=0, type=int, help='delta label stop condition between every update iteration interval.')
            parser.add_argument('--activation_delta', default=500, type=int, help='Start of delta check on labels assignments.')

        return parser

    def __init__(self, opt):
        """Initialize this model class
        Parameters:
            opt (Option class)-- training/test options



        """
        BaseModel.__init__(self, opt)  # call to initilization method of BaseModel

        self.init_DCEC()
        # specify training losses to print out.
        if opt.phase != 'test':
            self.loss_names = ['TOTAL', 'KL', 'REC'] if opt.phase == 'train' else ['REC']
            self.set_log_losses()

            # specify the models you want to save to the disk. The training/test scripts will call
            # <BaseModel.save_networks> and <BaseModel.load_networks>.
            self.model_names = ['E', 'D', 'CL'] if opt.phase == 'train' else ['E', 'D']
        else:
            self.model_names = ['E', 'D', 'CL']


        # Define the networks for DCEC and initialize them ( both netE-netD for the representation module)
        # and Clustering layer module.

        self.netE = init_net(net=Encoder(opt=opt), init_type=opt.init_type,
            init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.netD = init_net(net=Decoder(opt=opt), init_type=opt.init_type, init_gain=opt.init_gain,
            gpu_ids=self.gpu_ids)
        self.netCL = init_net(net=Clustering_Layer(opt=opt), init_type=opt.init_type, init_gain=opt.init_gain,
            gpu_ids=self.gpu_ids)
        # 0) Networks shapes input-output:
        self.networks_shapes = {**self.netE.module.network_info, **self.netD.module.network_info}

        if self.isTrain:
            # Define Loss Functions
            self.criterionPretrain = DCECLoss(phase='pretrain', reconstruction_mode=opt.rec_type)
            self.criterionTrain = DCECLoss(phase='train', reconstruction_mode=opt.rec_type, gamma=opt.gamma)
            # Initialize Optimizers Train/Pretrain; schedulers automatically created by function <BaseModel.setup>.
            self.optimizer_Pr = torch.optim.Adam(itertools.chain(self.netE.parameters(), self.netD.parameters()),
                                    lr=opt.lr_pr, betas=(opt.beta1, 0.999))
            self.optimizers.pretrain = self.optimizer_Pr
            self.optimizer_Tr = torch.optim.Adam(itertools.chain(self.netE.parameters(), self.netD.parameters(),
                                    self.netCL.parameters()), lr=opt.lr_tr, betas=(opt.beta1, 0.999))

            self.optimizers.train = self.optimizer_Tr

            # Directory overwriting:
            self.opt.path_man.set_dir(dir_to_extend="plots_dir", path_ext="reconstructed", force=True)  #reconstructed folder
            print('path_reconstructed :', self.opt.path_man.get_path(name='reconstructed_dir'))
            # metrics log
            self.metrics_names = {'avg_Si_score', 'Calinski-Harabasz score', 'Davies-Bouldin score'}
            self.set_metrics()
        else:
            # Test Task
            self.test_initialization()

    def test_initialization(self):
        """ Settings of the model's modules for the evaluation task """
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
                net.eval()

    def init_DCEC(self):
        """Init function for DCEC model:
        1) settings of pretrain's folders paths for the pretraining
        2) creation of the attribute self.name
        2) creation of the model folders
        4) creation of the attribute self.save_dir for the model class reference
        """
        if self.opt.phase == "train":
            # 1) Settings of pretrain's paths
            self.set_pretrain_folders()
            # 2) creation of the model name: "DCECModel_#clusters" if the phase is "train", "CAE_type" if the phase is "pretrain"
            self.name = self.__class__.__name__
            self.name = str().join([self.name, '_', str(self.opt.num_clusters)])
            # 3) creation of the model string for reference in the path Manager and later the model path/folder.
            string_model_dir = str().join([self.name, '_', 'dir'])
            self.opt.model_dir = string_model_dir
            # 4) creation of the model path/folder and subfolders for logs/weights/plots for the training phase.
            self.opt.path_man.change_phase(state="train")
            self.opt.path_man.initialize_model(model=self.name)
            self.save_dir = self.get_path_phase(name=string_model_dir)
        elif self.opt.phase == "pretrain":
            # 1) Settings of pretrain's paths
            self.set_pretrain_folders()
            # 2) creation of the model name: "DCECModel_#clusters" if the phase is "train", "CAE_type" if the phase is "pretrain"
            self.name = self.opt.AE_type
            # 3) creation of the model string for reference in the path Manager.
            string_model_dir = str().join(["model", '_', 'dir'])
            self.opt.model_dir = string_model_dir
        elif self.opt.phase == "test":
            # 2) creation of the model name: "DCECModel_#clusters" if the phase is "test"
            self.name = self.__class__.__name__
            self.name = str().join([self.name, '_', str(self.opt.num_clusters)])
            # 3) creation of the model string for reference in the path Manager and later the model path/folder.
            string_model_dir = str().join([self.name, '_', 'dir'])
            self.opt.model_dir = string_model_dir
            # 4) creation of the model path/folder and subfolders for logs/weights/plots for the testing and training phase.
            self.save_dir = self.get_path_phase(name='save_dir')
            self.opt.path_man.set_phase("train")
            self.opt.path_man.exp_name_assignment()
            self.opt.path_man.initialize_model(model=self.name)
            self.load_dir = self.get_path_phase(name=string_model_dir)
            self.opt.path_man.set_phase("test")
    def set_log_losses(self):
        """create the accuracy losses and logging file to store training losses"""
        # TODO correct the file extension of the loggers


        self.acc_losses = OrderedDict({name_losses: list() for name_losses in self.loss_names})
        self.log_losses = os.path.join(self.get_path_phase(name='logs_dir'), 'losses_{}_log_{}.txt'.format(self.opt.phase, self.name if self.opt.phase == "train" else self.opt.AE_type))
        first = 'epoch'
        for k in self.loss_names:
            first += ',' + k
        with open(self.log_losses, "w") as log_file:

            log_file.write('================ Training Loss ================\n' )
            log_file.write(first)
    def set_metrics(self):
        """create the accuracy losses and logging file to store training losses"""
        self.metrics_names = {'avg_Si_score', 'Calinski-Harabasz score', 'Davies-Bouldin score'}
        self.log_metrics = os.path.join(
            self.get_path_phase(name='logs_dir'), 'metrics_CVI_during_training_log.txt'
            )
        first = 'epoch'
        for k in self.metrics_names:
            first += ',' + k
        with open(self.log_metrics, "w") as log_file:

            log_file.write('================ Metrics Report ================\n' )
            log_file.write(first)

    def accumulate_losses(self):
        """Function to accumulate losses"""
        for name in self.loss_names:
            if isinstance(name, str):
                self.acc_losses[name].append(float(getattr(self, name + '_loss')))

    def reset_accumulator(self):
        """Reset the losses accumulator, creating a dictionary by name of loss function and empy list."""
        self.acc_losses = OrderedDict({name_losses: list() for name_losses in self.loss_names})

    def save_image_reconstructed(self, epoch):
        """ Function to plot and save the original images versus reconstructed images.
            Parameters:
                epoch (int): epoch of the representation"""

        n = min(self.x_batch.size(0), 2)
        path_reconstructed = self.opt.path_man.get_path(name='reconstructed_dir')
        path_epoch = os.path.join(path_reconstructed, 'reconstructed_images_epoch_{%d}') % (epoch)
        util_general.mkdir(path=path_epoch)

        for i in range(n):
            image = torch.cat([self.x_batch[i].to(self.device), self.x_hat[i].to(self.device)], 1)
            img = np.array(image.detach().cpu())
            img = 255.0 * img
            img = img.astype(np.uint8)
            img = Image.fromarray(img[0, :, :])
            if self.opt.dataset_name =='GLOBES':
                img.save(
                    fp=os.path.join(path_epoch, f"model_{self.__class__.__name__}_rec_IMG_epoch_{epoch}_IDpatient_{self.y_batch[i]}_{self.id_batch[i]}.tif")
                )
            else:

                img.save(
                    fp=os.path.join(path_epoch, f"model_{self.__class__.__name__}_rec_IMG_epoch_{epoch}_IDpatient_{self.y_batch[i]}_{self.id_batch[i]}.tif")
                )
    def print_metrics(self, epoch) :
        """print current epoch metrics calculated on console and save them in the log direc

               Parameters:
                   epoch (int) -- current epoch
               """
        if not np.unique(self.y_prediction).__len__() <= 1:
            print("Adding line to metrics result, epoch: ", epoch)
            with torch.no_grad():
                print("Labels assigned unique: ", np.unique(self.y_prediction))

                computed_metrics = metrics_unsupervised_CVI(Z_latent_samples=self.z_encoded_tot.detach().cpu(), labels_clusters=self.y_prediction)

            message = str(epoch)
            for v in computed_metrics.values():
                message += ','
                message += '%.3f ' % (v)
            print("".center(100, '|'))
            print(" Printing Metrics ".center(100, '|'))

            print("1) Average SI score", computed_metrics['avg_Si_score'], '\n',
                  "2) Calinski-Harabasz Score", computed_metrics['Calinski-Harabasz score'], '\n',
                  "3) 'Davies-Bouldin score'", computed_metrics['Davies-Douldin score'], '\n') if self.opt.verbose else self.do_nothing()  # print the message
            print("".center(100, '|'))
            with open(self.log_metrics, "a") as log_file:
                log_file.write('\n%s' % message)  # save the message
        else:
            print("Error only 1 label assigned to images")
    def print_current_losses(self, epoch, iters):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        print_message = 'EPOCH : %d  ; ' % (epoch)
        for k, v in self.acc_losses.items():
            print_message += f"{k} : %.3f  ; " % (self.opt.gamma * (np.sum(v) / (iters / self.opt.batch_size))) if k == 'KL' else f"{k} : %.3f  ; " % (
                        1.0 * (np.sum(v) / (iters / self.opt.batch_size)))
        print("".center(100, '*'))
        print(print_message[:-2])
        message = str(epoch)
        for k, v in self.acc_losses.items():
            message += ','
            message += '%.3f ' % (self.opt.gamma * (np.sum(v) / (iters / self.opt.batch_size))) if k == 'KL' else '%.3f ' % (
                        1.0 * (np.sum(v) / (iters / self.opt.batch_size)))
        # print the message
        with open(self.log_losses, "a") as log_file:
            log_file.write('\n%s' % message)  # save the message

    def print_current_losses_validation(self, epoch, iters):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = str(epoch)
        for v in self.acc_losses.values():
            message += ','
            message += '%.3f ' % (np.sum(v)/(iters/self.opt.batch_size))
        print(self.opt.verbose)
        print(message) # print the message
        with open(self.log_losses, "a") as log_file:
            log_file.write('\n%s' % message)  # save the message
    def set_input(self, input):
        """Unpack input data from dataloader and perform eventually preprocessing
        Parameters:
            input (torch.Tensor ): batch data, or single image.

        """
        tensor = input[0]
        self.x_batch = tensor.type(torch.FloatTensor).to(self.device) # copy
        self.y_batch = input[1]
        self.id_batch = input[2] if self.opt.dataset_name else None
        # copy

    def encode(self):
        """Run encoding pass"""
        with torch.no_grad():
            return self.netE(self.x_batch)
    def set_target_p_batch(self, ind):
        """Settings the targe target probability respect 1 batch of data
        Parameters:
            ind (int): index of the batch during the iteration of the DataLoader
        """
        self.target_prob_batch = self.target_prob[ind * self.opt.batch_size::] if (ind + 1) * self.opt.batch_size > self.opt.dataset_size else self.target_prob[ind * self.opt.batch_size:
                                                                                          (ind + 1) * self.opt.batch_size]

    def get_predictions_probabilities(self, z_latent):
        with torch.no_grad():

            q_ij = self.netCL(z_latent)
            labels_y = q_ij.argmax(1).detach().cpu()  # selecting labels
            return [labels_y, q_ij.detach().cpu()]

    def save_actual_labels(self, dictionary_data, Dataset_z, Dataset_qij):
        print("".center(100, '|'))
        print(" Saving of clusters labels file ".center(100, '|'))
        print("".center(100, '|'))
        # Creation of the labels directory
        labels_file = os.path.join(self.opt.path_man.get_path('labels_dir'), 'data_clusters_labels_K_%s_.xlsx' %format(str(self.opt.num_clusters)))
        data = pd.DataFrame.from_dict(dictionary_data)
        data.set_index('img ID')
        with pd.ExcelWriter(labels_file) as writer:
            data.to_excel(writer, sheet_name='Data_complete')


        # Datasets Directory
        Datasets_file = os.path.join(self.opt.path_man.get_path('labels_dir'),'datasets_z_q_K_%s'%format(str(self.opt.num_clusters)) )
        np.savez_compressed(Datasets_file, Z_dataset=Dataset_z, Q_Dataset=Dataset_qij)
        print('Information and Datasets have been saved as .xlxs and .npz files in labels directory: ', self.opt.path_man.get_path('labels_dir'))

        print("".center(100, '|'))
        print("".center(100, '|'))

    def update_target(self, dataloader, indexing):
        """Update target probability every # train iterations
        Returns:
            delta_label (float): is the label assignment difference between a fixed interval of training iterations. I's a float if the delta label parmeter is set not equal to zero, mean

        """
        with torch.no_grad():
            output = self.compute_encoded(dataloader=dataloader)
            self.z_encoded_tot = output['z_latent']
            self.x_tot = output['x_out']
            # probabilities computed for each samples to belong to each n_clusters.

            q_ij = self.netCL(self.z_encoded_tot)

            self.target_prob = target_distribution(q_ij=q_ij)  # set target distribution

            # Predictions labels for each sample

            y_pred = q_ij.argmax(1).detach().cpu().numpy()

            # Dictionary to save
            dict_to_save = {
                'img ID': [id + '_' + img_id for id, img_id in zip(output['id_patient'], output['image_id'])],
                'patient ID': output['id_patient'],
                'image_id': output['image_id'],
                'clusters_labels': list(y_pred)}
            # Datasets to save
            Datasets = {
                        'Dataset_z': self.z_encoded_tot.numpy().copy(),
                        'Dataset_qij': q_ij.detach().cpu().numpy()
                        }

            # Save the Excel file for data with labels

            self.save_actual_labels(dictionary_data=dict_to_save, **Datasets)
            uniques, counts = np.unique(np.array(y_pred), return_counts=True)
            print("".center(100, '|'))
            print(" CLUSTERING INFORMATIONS ".center(100, '|'))
            print(f' 1) The list of unique labels assigned: {uniques} \n '
                  f' 2) Uniques distribution:  {counts}')
            # For the first updating the labels are given by k-means fitting algorithm
            y_pred_last = np.copy(self.y_prediction)
            # Setting of the new labels assigned to samples
            self.y_prediction = np.copy(y_pred)
            # check stop criterion
            if self.opt.delta_check:
                # Check stop condition if the parameter is set in option.
                y_pred_last = y_pred_last[indexing[0]]
                delta_label = np.sum(np.array(y_pred) != y_pred_last).astype(np.float32) / \
                              y_pred.shape[0]
                print(
                      "3) Delta label is : ", delta_label)

                print("".center(100, '|'))
                return delta_label < self.opt.delta_label
            else:
                return False

    def set_pretrain_folders(self):
        """Set the pretrain folders in the path manager."""
        print("INFO: settings pretrain paths into DCEC models.")
        self.opt.path_man.set_phase(phase="pretrain")
        self.opt.path_man.exp_name_assignment()
        self.opt.path_man.initialize_model()

    def load_model_pretrained(self):
        """Ths function guarantees to load the weights pretrained for Encoder and Decoder"""
        load_suffix = 'iter_%s' % (str(self.opt.load_iter)) if self.opt.load_iter > 0 else 'latest'
        return self.load_networks(epoch=load_suffix, path_ext=self.get_path_phase("weights_dir", phase="pretrain"))

    def load_model_trained(self):
        """Ths function guarantees to load the weights trained for Encoder, Decoder and Clustering layer."""
        load_suffix = 'iter_%s' % (str(self.opt.load_iter)) if self.opt.load_iter >0 else 'latest'
        return self.load_networks(epoch=load_suffix, path_ext=self.get_path_phase("weights_dir", phase="train"))

    def compute_encoded(self, dataloader):
        import time
        x_out = None
        y_out = None
        id_out = None
        z_latent_out = None
        print('INFO: encoding all data on course...')
        time_0 = time.time()
        time_f = time.time()
        with tqdm(dataloader, unit="batch", desc="Progress bar ENCODING DATA...") as tqdm_encoding:
            for data in tqdm_encoding:
                self.set_input(data)

                z_latent_batch = self.netE(self.x_batch)  # pass batch of samples to the Encoder
                print(torch.cuda.current_device())
                print(
                    "Outside: input size", self.x_batch.device,
                    "output_size", z_latent_batch.device
                )
                # ----------------------------------
                # Concatenate z latent samples and x samples together
                x_out = np.concatenate((x_out, data[0]), 0) if x_out is not None else data[0]
                y_out = np.concatenate((y_out, data[1]), 0) if y_out is not None else data[1]
                id_out = np.concatenate((id_out, data[2]), 0) if id_out is not None else data[2] if len(data)>1 else None
                z_latent_out = np.concatenate((z_latent_out, z_latent_batch.cpu().detach().numpy()), 0) if z_latent_out is not None else z_latent_batch.cpu().detach().numpy()
        print("INFO: ---> Encoding done in : ", (time_f - time_0) / 60, '( min. )')
        return {'id_patient': list(y_out), 'z_latent': torch.from_numpy(z_latent_out), 'x_out': torch.from_numpy(x_out), 'image_id': list(id_out)} if len(data)>1 else {'y_out': list(y_out), 'z_latent': torch.from_numpy(z_latent_out), 'x_out': list(x_out)}

    def prepare_training(self, dataset):
        """Procedure to start the training of Deep Convolutional Embeddings Clustering model
        i) Initialize clusters centers with KMEANS clustering algorithm.
        ii) extract clusters centers from fitted kmeans and put them in to <ClusteringLayer>.
        """
        # i) Fitting clusters centers with k-means
        self.x_tot, self.kmeans, self.y_prediction = kmeans(model=self, dataloader=dataset.dataloader_big_batch, opt=self.opt)
        # ii) Set the centers parameters in the ClusteringLayer Module
        cluster_centers = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)
        # ADD clusters parameters in DCEC:
        with torch.no_grad():
            # initialise the cluster centers
            self.netCL.module.set_centroid_centers(cluster_centers)
        # ______________________________________________________________________________________________________________
        #  Change folders
        self.setup(opt=self.opt)  # reconfigure optimizers
        # creation of the labels directory in each models savings path
        self.opt.path_man.set_dir(dir_to_extend=f'DCECModel_{self.opt.num_clusters}_dir', path_ext='labels', name_att='labels')
        self.opt.path_man.change_phase(self.opt.phase)  # change phase dictionary for reference
    def forward(self):
        """Run forward pass train/pretrain"""

        # Autoencoding images
        self.z_encoded = self.netE(self.x_batch)
        self.x_hat = self.netD(self.z_encoded)
        # Passing the batch through the Clustering layer
        if self.opt.phase == "train":
            self.qij_assignment = self.netCL(self.z_encoded)
    def backward_DCEC(self):
        """Calculate the loss for DCEC in Train/Pretrain"""
        x_batch = self.x_batch.to(self.device) # Pass the images q_ij to device.

        if self.opt.phase == "train":
            # Backward pass for CAE module in DCEC model for the pretraining phase.
            self.TOTAL_loss, self.KL_loss, self.REC_loss = self.criterionTrain(x_batch, self.x_hat, self.target_prob_batch,
                self.qij_assignment)
            self.TOTAL_loss.backward()
        elif self.opt.phase == "pretrain":
            # Backward pass for CAE module in DCEC model for the pretraining phase.
            self.REC_loss = self.criterionPretrain(x_batch, self.x_hat)
            self.REC_loss.backward()
    def compute_metrics(self):
        pass

    def set_p_target(self, z):
        """Set distribution p for KL divergence loss"""
        self.p_target = self.get_assignment(z)

    def get_assignment(self, x):
        """Compute qij_assignment on batch image x"""
        z_encoded = self.netE(x)
        return self.netCL(z_encoded)
    def compute_labels(self, z_encoded):
        """Compute the labels for each samples in batch image x"""
        qij = self.netCL(z_encoded)
        labels = qij.argmax(1).detach().cpu()
        return labels
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        phase = self.opt.phase
        # Pretrain
        if self.opt.phase == "pretrain":
            self.set_requires_grad([self.netCL], False)  # Clustering module don't need to be updated
            # during pretraining phase
            # forward
            self.forward()
            # DCEC backward pass
            self.optimizers[phase].zero_grad()  # set netE and netD to zero
            self.backward_DCEC()           # calculate gradients for netD and netE
            self.optimizers[phase].step()       # update weights
        elif self.opt.phase == "train":
            # training
            # forward
            self.forward()
            # DCEC backward pass
            self.optimizers[phase].zero_grad()  # set netE, netD and Clustering layer to zero
            self.backward_DCEC()           # calculate gradients for netD and netE
            self.optimizers[phase].step()       # update weights













