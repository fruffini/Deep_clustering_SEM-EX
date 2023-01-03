import functools
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from util.shape_functions import *
from util.util_general import *
from easydict import EasyDict as edict
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts



def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: q_ij | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    """
    if norm_type == 'q_ij':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
def init_models(net, init_gain=0.02, init_type='normal', gpu_ids=list):
    """init_models
    Parameters:
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns:
        netE-netD models


    """
    return init_net(net=net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer: network's optimizer
        opt : (opt class), store al the experimenst options, has to be a subclass of Opti
        For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
        and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
        For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
        See https://pytorch.org/docs/stable/optim.html for more details.
    Returns:
        Scheduler (object)
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    elif opt.lr_policy == 'cosine-warmup':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=opt.n_epochs + opt.n_epochs_decay, cycle_mult=1.0, max_lr=opt.lr_tr, min_lr=opt.lr_tr*0.001, warmup_steps=opt.n_epochs, gamma=0.5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=list):
    """Initialize a network:
    1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    print(gpu_ids)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)
    return net
def configure_optimizers(model, opti_mode, hyperparameters):
    """
    :return:  Optimizer
    :rtype:   (torch.optim.Optimizer)
    """

    optimizer = None

    lr = hyperparameters["learning_rate"]
    if opti_mode == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=0,
            initial_accumulator_value=0, eps=1e-10)
    elif opti_mode == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer
def target_distribution(q_ij):
    """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param q_ij: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float"""
    weight = (q_ij ** 2) / torch.sum(q_ij, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

##############################################################################
# Classes
##############################################################################


class Identity(nn.Module):
    """
    Identity model function: return the tensor itself without the
    """
    def forward(self, x):
        return x


class DCECLoss(nn.Module):
    """Define the loss needed to train the DCEC, choosing which phase is --("pretrain", "train").
    It can be choose wich kind of reconstruction loss has to be used in the experiment run defining
    reconstruction_mode .
    """

    def __init__(self, phase='pretrain', reconstruction_mode='bce', gamma=0.0):
        """ Initialize the DCECLoss class.

        Parameters:
            phase (str): training phase of DCEC algorithm.
            reconstruction_mode (str): reconstruction loss used in the experiment ("mse","bce","bce_l")
            gamma (double): parameter to bilance reconstruction and KL divergence losses
                    Loss = L_rec + gamma * K_L_divergence"""
        super(DCECLoss, self).__init__()
        self.phase = phase
        self.reconstruction_mode = reconstruction_mode
        self.gamma = gamma
        self.reconstruction_loss = self.reconstruction_mode_loss()
        if self.phase == "train":
            self.kl_loss = self.kl_divergence_loss()
            return
        elif self.phase == "pretrain":
            return
        else:
            raise NotImplementedError('The phase : %s is not implemented' % phase)

    def reconstruction_mode_loss(self):
        """ Initialize and compute the reconstruction loss function.
        Parameters:
            mode (str): type of reconstruction loss function.
        Returns:
            Loss (object)
        """
        Loss = None
        if self.reconstruction_mode == "mse":
            Loss = nn.MSELoss(reduction='mean')
        elif self.reconstruction_mode == "bce":
            #Binary cross entropy to compute the reconstruction loss function.
            Loss = lambda x, x_hat: -torch.sum(
                x * torch.log(torch.clamp(x_hat, min=1e-10)) + (1 - x) * torch.log(torch.clamp(1 - x_hat, min=1e-10)),1)
        elif self.reconstruction_mode == "bce_l":
            Loss = nn.BCEWithLogitsLoss()
        return Loss

    def kl_divergence_loss(self):
        """
        Calculate Kullback-Leibler divergence D_kl(P || Q)
        Parameters:
            p: target probability distribution
            q: t-Student probability distribution computed from clustering layer output.
        """
        kl_loss = nn.KLDivLoss(size_average=False)
        #kl_loss = F.kl_div(q.log(), p)
        return kl_loss

    def __call__(self, x, x_hat, p_target=None, q_assigned=None):
        """Calculate loss for pretrain/train phase of DCEC algorithm.

            Parameters:
                x (tensor) - - original image
                x_hat (tensor) : reconstructed image by Autoencoder
                p_target (numpy.array) : target probability distribution.
                q_assigned (numpy.array) : label t-Student probability distribution.
            Returns:
                the calculated loss.
            """
        rec_loss = torch.mean(self.reconstruction_loss(x, x_hat)) # rec loss
        if self.phase == "pretrain":
            return rec_loss
        elif self.phase == "train":
            kl_loss = self.kl_loss(q_assigned.log(), p_target)  # kl loss
            tot_loss = rec_loss + self.gamma * kl_loss  # DCEC training Loss
            return tot_loss, kl_loss, rec_loss


class Encoder(torch.nn.Module):
    """Define netE network"""

    def __init__(self, opt):
        """
        Create netE model.
        opt (Namespace): options to initialize the netE
        """
        super(Encoder, self).__init__()
        self.opt = opt  # Options for the experiment
        self.arch = self.set_architecture_parameters()  # Architecture
        self.network_info = dict()
        self.convolutional_layers = self._build_conv_encoder()  # Convolutional layer construction
        # Set_information:
        self.encoder_shapes = self.set_encoder_information()
        # ______________________________________________________________________________________________________________
        self.embedded = nn.Linear(in_features=self.features_flatten, out_features=self.latent_dimension)
        self.network_info["shape_{}".format("embedding_H")] = [self.features_flatten, self.latent_dimension]
        opt.encoder_shapes = edict(self.encoder_shapes)  # Add convolutional last shape to Option-Class main routine.
    def set_architecture_parameters(self):
        """ Setting the parameters to build the netE structure
        Returns:
            config layers (dict): the dictionary with every sinngle layer parameters settings.
        """
        self.latent_dimension = self.opt.embedded_dimension
        return edict(load_config(self.opt.AE_cfg_file, self.opt.config_dir)[self.opt.AE_type])

    def _build_conv_encoder(self):
        """
        Building of the convolutinal Downsampling component of the netE. This function provides the organization
        of the
        model and the definition of the bottleneck dimension.
        :return:   PyTorch Sequential model,  struttura convolutional netE;
        :rtype:   (torch.nn.Sequential)
        """
        # inizializzazione variabili:
        input_shape = self.opt.image_shape  # image shape input
        in_channels = self.opt.input_nc
        current_shape = (input_shape,)*2
        features_channels = self.arch.features_channels
        kernels = self.arch.kernels
        strides = self.arch.strides
        paddings = self.arch.paddings

        # build  the encoder:
        conv_sets = list()

        for layer in range(features_channels.__len__()):
            out_channels = features_channels[layer]
            kernel = tuple(kernels[layer])
            stride = tuple(strides[layer])
            padding = tuple(paddings[layer])

            # Build a set of layers composed as:
            # - Convolutional
            # - Leaky ReLU / Relu
            conv_sets.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                    padding=padding))
            # Batch Normalization
            if self.opt.norm != "none":
                conv_sets.append(get_norm_layer(norm_type=self.opt.norm))

            conv_sets.append(nn.Dropout(p=self.opt.dropout))
            conv_sets.append(nn.LeakyReLU(negative_slope=self.opt.leaky_nslope) if self.opt.leaky else nn.ReLU())
            precedent_shape = current_shape
            # compute the new shape:
            current_shape = compute_output_shape(current_shape=current_shape, kernel_size=kernel, stride=stride,
                padding=padding)

            self.network_info["shape_{}_{}".format("convolutional_layers", layer + 1)] = [
                (in_channels,) + precedent_shape, (out_channels,) + current_shape]
            # the output == input: next step.
            in_channels = out_channels

        # Una volta creato il modello Sequential desiderato lo si scrive nella classe encoder.
        return nn.Sequential(*conv_sets)

    def set_encoder_information(self):
        """
        Returns:
            encoder layers architecture information, needed to build the mirror decoder.
        """

        shapes = list()
        for step, (key, value) in enumerate(self.network_info.items()):
            if step == 0:
                shapes.append(value[0])
            shapes.append(value[1])
        self.features_flatten = shapes[-1][0] * np.prod(shapes[-1][1:])
        return {"features_in": self.features_flatten, "encoder_shapes": shapes, }

    def forward(self, x):
        """
        Forward call for netE
        :param x: Tensor Batch, ( N, C, H, W)
        :return: Tensor Batch, z_latent (H,1)
        """
        x = self.convolutional_layers(x)
        x = torch.flatten(x, start_dim=1)
        z_latent = self.embedded(x)  # Spazio latente autoencoder
        return z_latent


class Decoder(nn.Module):
    def __init__(self, opt, **kwargs):

        super(Decoder, self).__init__(**kwargs)

        # La caretteristica principale è che il decoder viene costruito con la stessa architettura dell'netE
        # Ma chiaramente con layer de-convolutional, quindi che riducono la dimensionalità e aumentano la dimensione
        # HxW delle q_ij di immagini
        # ______________________________________________________________________________________________________________
        #################
        # netD Model #
        #################
        self.opt = opt
        self.arch = self.set_architecture_parameters()
        self.shapes = opt.encoder_shapes.encoder_shapes
        self.network_info = dict()
        self.features_in_dec = opt.encoder_shapes.features_in
        self.deembeddings = nn.Linear(in_features=opt.embedded_dimension, out_features=self.features_in_dec)
        self.transpose_layers = self._build_transpose_convolution_layers()



        """if self.activation == "sigmoid":
            hiddens_layers.append(nn.Sigmoid())
        elif self.activation == "tanh":
            hiddens_layers.append(nn.Tanh())"""

    def set_architecture_parameters(self):
        """ Setting the parameters to build the netE structure
        Returns:
            config layers (dict): the dictionary with every sinngle layer parameters settings.
        """
        self.latent_dimension = self.opt.embedded_dimension
        return edict(load_config(self.opt.AE_cfg_file, self.opt.config_dir)[self.opt.AE_type])

    def _build_transpose_convolution_layers(self):
        """Bulding of convolutional transpose component of netD network.

            :return:   PyTorch Sequential model,  struttura deconvolutinal netD.
            :rtype:   (torch.nn.Sequential)


        """
        # ______________________________________________________________________________________________________________
        # DECODER : CONV TRANSPOSE CONSTRUCTION
        features_channels = self.arch.features_channels
        kernels = self.arch.kernels
        strides = self.arch.strides
        paddings = self.arch.paddings
        encoder_shapes = self.shapes

        conv_sets = list()
        in_channels = features_channels[-1]  # To build the netD we need to reverse the parameters of the NET,
        # starting from last to first ones.

        # Reverse iteration over layers
        for layer in range(features_channels.__len__() - 1, -1, -1):

            out_channels = self.opt.input_nc if layer == 0 else features_channels[layer - 1]
            kernel = kernels[layer]
            stride = strides[layer]
            padding = paddings[layer]

            # Computing the output padding to apply at the image of current_shape to obtain the target shape
            # applying transpose convolution.

            current_shape = encoder_shapes[layer + 1][1:]
            target_shape = encoder_shapes[layer][1:]

            output_shape = compute_transpose_output_shape(current_shape=current_shape, kernel_size=kernel,
                stride=stride, padding=padding)

            # compute the output padding
            output_padding = compute_output_padding(output_shape, target_shape)

            # add information Network:
            self.network_info["shape_{}_{}".format("deconv_layer", features_channels.__len__() - layer)] = [
                (in_channels,) + current_shape, (out_channels,) + target_shape]
            if layer == 0:
                # Final Layers Set ConvolutionalTranspose - Relu/Leaky ReLU -Convolutional - Sigmoid
                conv_sets.append(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel,
                        stride=stride, padding=padding, output_padding=output_padding), )

                if self.opt.activations == "sigmoid":
                    conv_sets.append(nn.Sigmoid())
                elif self.opt.activations == "tanh":
                    conv_sets.append(nn.Tanh())
            else:
                # Intermediate Layers set of ConvolutionalTranspose - Dropout - Relu/Leaky ReLU - (BatchNorm)
                conv_sets.append(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel,
                        stride=stride, padding=padding, output_padding=output_padding))
                conv_sets.append(nn.Dropout(p=self.opt.dropout))
                conv_sets.append(nn.LeakyReLU(negative_slope=self.opt.leaky_nslope) if self.opt.leaky else nn.ReLU())
                # Batch Normalization
                if self.opt.norm != "none":
                    conv_sets.append(get_norm_layer(norm_type=self.opt.norm))
            # the output channels of the current layer becomes the input channels of the next layer
            in_channels = out_channels

        # create a Sequential model and return it (* asterisk is used to unpack the list)
        return nn.Sequential(*conv_sets)

    def forward(self, z):
        # run the latent vector through the "input decoder" layer
        """Forward call from latent code of the decoder (H), through the deconvolutional component of the decoder,
        and return the x_hat reconstructed image.


        Parameters:
            z (torch.Tensor): latent code vector
        Returns:
            x_hat (torch.Tensor): reconstructed original image
            """
        deembeddings = self.deembeddings(z)
        # convert back the shape that will be fed to the decoder
        height = int(np.sqrt(deembeddings.shape[1]/self.arch.features_channels[-1]))
        width = height

        # run through the decoder
        decoder_input = deembeddings.view(-1, self.arch.features_channels[-1], height, width)  # de_flattening
        x_hat = self.transpose_layers(decoder_input)
        return x_hat


class Clustering_Layer(nn.Module):
    def __init__(self, opt, clusters_centers=None):
        """
                Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
                where the Student's t-distribution is used measure similarity between feature vector and each
                cluster centroid.
                Parameters:
                    opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
                    alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
                    clusters_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(Clustering_Layer, self).__init__()
        self.opt = opt
        self.embeddings_dimension = opt.embedded_dimension
        self.num_Clusters = opt.num_clusters
        self.alpha = opt.alpha
        self.clusters_centers = clusters_centers
        if clusters_centers is None:
            initial_clusters_centers = torch.zeros(self.num_Clusters, self.embeddings_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_clusters_centers)  # Xavier inizialization of weights
        else:
            initial_clusters_centers = clusters_centers
        self.weight = nn.Parameter(initial_clusters_centers)
    def set_centroid_centers(self, clusters_centers):
        """Set clusters centers into Clustering layers as training parameters Parameter
        Parameters:
              clusters_centers(torch.Tensor): input centroids of K_means to calculate the Assignation to the clusters computing
               t-Student probability.
        """

        self.state_dict()["weight"].copy_(clusters_centers)
    def forward(self, batch):
        """
                Compute the soft assignment for a q_ij of feature vectors, returning a q_ij of assignments
                for each cluster. alpha is the degree of freedom
                :param batch: FloatTensor of [q_ij size, embedding dimension]
                :return: FloatTensor [q_ij size, number of clusters]
        """
        norm_ = torch.sum((batch.unsqueeze(1) - self.weight) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_ / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

