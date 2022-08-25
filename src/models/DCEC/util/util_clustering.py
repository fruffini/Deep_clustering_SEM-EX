"""Unsupervised Metrics script

This script provides the main functions needed to evaluate the behavior of the model during training or post training.
First, in this script is implemented the function needed to run the clustering algorithm "K-means" over n-samples complete dataset represented in the latent space (for other
information go to "k-means" method's docstring)
Internal Validity Index Metrics ( NO-label-needed ) :
    1) Silohuettes Scores: Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1.
        1: Means clusters are well apart from each other and clearly distinguished. ( GOOD )
        0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.
        -1: Means clusters are assigned in the wrong way.   ( BAD )
        To obtain a general behaviour of clusters the average could be applied to singles silhouette scores.
        LINK: https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c
    2) Calinski-Harabasz (CH) Index: (introduced by Calinski and Harabasz in 1974) can be used to evaluate the model when ground truth labels are not known
        where the validation of how well the clustering has been done is made using quantities and features inherent to the dataset.
        The CH Index (also known as Variance ratio criterion) is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
         Here cohesion is estimated based on the distances from the data points in a cluster to its cluster centroid and separation is based on the distance of the cluster
         centroids from the global centroid. CH index has a form of (a . Separation)/(b . Cohesion) , where a and b are weights.
         1) +inf: Higher value of CH index means the clusters are dense and well separated, although there is no “acceptable” cut-off value.
         2) 0: the distance between clusters is not significant.
         LINK: https://www.geeksforgeeks.org/calinski-harabasz-index-cluster-validity-indices-set-3/
    3) Davies-Bouldin index (DB): (named after its creators, David Davies and Donald Bouldin) quantifies the average separability of each cluster from its nearest counterpart.
        It does this by calculating the ratio of the within-cluster variance (also called the scatter) to the separation between cluster centroids.
        If we fix the distance between clusters but make the cases within each cluster more spread out, the Davies-Bouldin index will get larger.
        Conversely, if we fix the within-cluster variance but move the clusters farther apart from each other, the index will get smaller.
        1) Ideal: In theory, the smaller the value (which is bounded between zero and infinity), the better the separation between clusters.
        LINK:  https://livebook.manning.com/concept/r/davies-bouldin-index

"""
import time
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def metrics_unsupervised_CVI(Z_latent_samples, labels_clusters):
    """
    Functions to compute unsupervised internal validation metrics.
    Parameters:
        Z_latent_samples (torch.Tensor) : encoded samples in the embedded space of DCEC.
        labels_clusters (torch.Tensor) : label associated for each sample that indicates in which cluster the sample belong to.
    Returns
    -------
    dict of scores (dict):
        key : type
        'avg_Si_score' : float
            Mean Silhouette Coefficient for all samples.
        'Calinski-Harabasz score' : float
            The resulting Calinski-Harabasz score.
        'Davies-Douldin score' : float
            The resulting Davies-Bouldin score.
    ----------
    """
    # 1) Silhouette_Score:
    Si_score = silhouette_score(Z_latent_samples, labels_clusters)
    # 2) Calinski_Harabasz_score:
    CH_score = calinski_harabasz_score(Z_latent_samples, labels_clusters)
    # 3) davies_bouldin_score:
    DB_score = davies_bouldin_score(Z_latent_samples, labels_clusters)
    return {'avg_Si_score': Si_score, 'Calinski-Harabasz score': CH_score, 'Davies-Douldin score': DB_score}


def kmeans(model, dataloader, opt):
    """
    K-means algorithm trained on samples represented Autoencoder latent space.
    Article: MacQueen, J. (1967). Classification and analysis of multivariate observations. In 5th Berkeley Symp. Math. Statist. Probability (pp. 281-297).
    link: https://www.cs.cmu.edu/~bhiksha/courses/mlsp.fall2010/class14/macqueen.pdf
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
    time_kmeans_0 = time.time()
    prediction = km.fit_predict(output_array)
    time_kmeans_f = time.time()
    print("Kmeans fitted on data \n Time needed for fitting", (time_kmeans_f-time_kmeans_0)/60, '( min. )')
    return x_out, km, prediction



def target_distribution(batch):
    """
        Compute the target distribution p_ij, given the q_ij (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [q_ij size, number of clusters] Tensor of dtype float
        :return: [q_ij size, number of clusters] Tensor of dtype float"""
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
