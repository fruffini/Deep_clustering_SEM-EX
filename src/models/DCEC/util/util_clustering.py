import numpy as np
import torch
from sklearn.cluster import KMeans


def kmeans(model, dataloader, opt):
    """
    K-means algorithm trained on samples represented Autoencoder latent space.
    :param model (<DCECmodel>):
    :param data: the same dataset used to pretrain the Autoencoder.
    :param opt (Option class): stores all the experiment flags; needs to be a subclass of BaseOptions
    :return: km (KMeans): returns the kmean algorithm trained on samples from dataloader represented in latent space.
    """
    print('INFO : ---> Initializing cluster centers with k-means.')
    km = KMeans(n_clusters=opt.num_clusters, n_init=100)
    output_array = None
    x_out = None
    for data in dataloader:
        model.set_input(data)
        z_latent_batch = model.encode()  # pass batch of samples to the Encoder
        # ----------------------------------
        # Concatenate z latent samples and x samples together
        x_out = np.concatenate((x_out, data[0]), 0) if x_out is not None else data[0]
        output_array = np.concatenate((output_array, z_latent_batch.cpu().detach().numpy()), 0) if output_array is not None else z_latent_batch.cpu().detach().numpy()
    # ----------------------------------
    # Fit k-means algorithm on concatenated samples and predict labels
    print("Kmeans fitting on course")
    prediction = km.fit_predict(output_array)
    print("Kmeans fitted on data")
    return x_out, km, prediction

def target_distribution(batch):
    """
        Compute the target distribution p_ij, given the q_ij (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [q_ij size, number of clusters] Tensor of dtype float
        :return: [q_ij size, number of clusters] Tensor of dtype float"""
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
