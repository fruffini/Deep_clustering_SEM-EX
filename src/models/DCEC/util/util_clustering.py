import time
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI


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


def weighted_var_and_var(values: object, weights: object) -> object:
    """
    Return the weighted average and traditional variance of values.


    values, weights -- Numpy ndarrays with the same shape.
    """
    mean = np.mean(values)
    # Fast and numerically precise:
    variance_w = np.average((values - mean) ** 2, weights=weights)
    variance = np.var(values)
    return variance, variance_w


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.00001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))



def compute_probabilities_variables(labels, probability, ids, id_dict, threshold=90):
    # Cerco le strnghe associate ad ogni paziente:
    patients = np.unique(ids)
    weights_k = np.bincount(labels)
    weights_k = weights_k[weights_k != 0]
    # Salvo il numero reale labels assegnate = numero di K effettivo.
    real_K = np.unique(labels)
    # Prendo le probilità di assegnazione delle labels, e ne calcolo il valore massimo:
    saved = probability.max(1)
    # Inserisco un threshold di confidenza.
    th = threshold
    # Prendo i valori massimi di probabilità, e ne calcolo il valore medio:
    max_values = [el.item() for el in saved[0].detach()]
    mean_max_probabilities = np.mean(max_values)
    # Seleziono sia le lebal che gli indici rispetto al threshold.
    indices_th = [max_ > th for max_ in max_values]
    labels_th = labels[indices_th]
    ids_th = ids[indices_th]
    mean_max = mean_max_probabilities
    prototypes_for_cluster = list()
    for label in real_K:
        index_for_label_th = [l == label for l in labels_th]
        ids_cluster_th = ids_th[index_for_label_th]
        number_of_prototypes = np.unique(ids_cluster_th).size
        prototypes_for_cluster.append(number_of_prototypes)
    mean_n_prototypes = np.average(prototypes_for_cluster, weights=weights_k)

    # Normalized Mutual information Score
    id_int_array = np.array([id_dict[id] for id in ids])
    Mutual_information_criterion = NMI(labels_true=id_int_array, labels_pred=labels)

    # Report print probabilities
    print('-----------------Report Probabilities Debugger ---------------------')
    print('ids: ', ids)
    print('labels_th: ', labels_th)
    print('ids_th: ', ids_th)
    print('max_values: ',max_values)
    print('Max values mean: ', mean_max_probabilities)
    print('--------------------------------------------------------------------')


    return mean_max, mean_n_prototypes, Mutual_information_criterion, indices_th, prototypes_for_cluster

def compute_GINI(list_distribution):
    gini_index_for_t = [gini(element[np.nonzero(element)].astype(np.float64)) for element in np.array(
        list_distribution)]
    cumulative_gini = np.sum(gini_index_for_t)
    return gini_index_for_t, cumulative_gini



def calc_Delta_metrics(matrix, N01=False):
    matrix = np.array(
        [list((matrix[:, i] - np.min(matrix[:, i])) / (np.max(matrix[:, i]) - np.min(matrix[:, i]))) for
         i in range(matrix.shape[1])]).transpose() if N01 else matrix.copy()
    D_matrix = np.array(
        [list((matrix[i + 1, :]) - (matrix[i, :])) for i in range(matrix.shape[0] - 1)])
    D_sum_matrix = D_matrix.sum(1)
    return matrix, D_matrix, D_sum_matrix



def TF_Variances_ECF(z_, labels, ids):
    patients = np.unique(ids)
    N_patients = len(patients)
    count_by_cluster = np.bincount(labels)
    K = len(count_by_cluster)
    index_clusters = np.arange(0, K)
    X_for_patient = list()
    labels_for_patient = list()
    # Variances: rispetto al numero di slices di t (t_t) e al numero di slices di k (t_k).
    Var_SF = list()
    Var_SF_weighted = list()
    list_Number_of_element_over_t = list()

    for patient in patients:
        ind = [patient == id for id in ids]
        X_for_patient.append(z_[ind])
        labels_for_patient.append(labels[ind])

    for t in range(N_patients):
        # COMPUTE OVER t:
        # Dataset for patient ( Tensor of slices for patient ID )
        X_ = X_for_patient[t]
        # Distribuzioni del paziente nei clusters:
        list_Number_of_element = list()
        list_SF = list()
        for k in np.unique(labels):

            # indicizzazione per cercare le immagini correlate con un certo cluster:
            select_t = [label.item() == k for label in labels_for_patient[t]]
            if not len(X_[select_t]) == 0 or True:
                # frequenza tra il numero di slices del paziente nel cluster k e il numero totale di slices del
                # paziente t:
                sf_t_t = len(X_[select_t]) / labels_for_patient[t].shape[0]
                # salvo tutti i valori in delle liste, scalabili successivamente con tensori o array:

                list_SF.append(sf_t_t)
                list_Number_of_element.append(len(X_[select_t]))
        Var ,Var_w = weighted_var_and_var(np.array(list_SF), np.array(list_Number_of_element))

        # Varianza normale e pesata:
        Var_SF_weighted.append(Var_w)
        Var_SF.append(Var)

        # Distribuzione di slices paziente over clusters:
        list_Number_of_element_over_t.append(list_Number_of_element)

    return Var_SF, Var_SF_weighted, list_Number_of_element_over_t

def target_distribution(batch):
    """
        Compute the target distribution p_ij, given the q_ij (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [q_ij size, number of clusters] Tensor of dtype float
        :return: [q_ij size, number of clusters] Tensor of dtype float"""
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
