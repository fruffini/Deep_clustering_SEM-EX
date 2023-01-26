import math
import time
from copy import deepcopy
from typing import Tuple, Dict, Any, List

import pandas as pd
from easydict import EasyDict as edict
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics.cluster import adjusted_mutual_info_score as ADJ_NMI
from tqdm import tqdm
import itertools
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


class Metrics_CCDC(object):
    def __init__(self, opt):
        self.opt = opt
        self.labels_for_each_K = edict()

    def add_new_Clustering_configuration(self, labels_info: pd.DataFrame) -> None:
        self.labels_for_each_K["K_{0}".format(self.opt.num_clusters)] = labels_info

    def compute_CCDC(self):
        """Computing Clustering"""
        try:

            all_Ks = [int(key.split('_')[1]) for key in self.labels_for_each_K.keys()]
            # Output Matrices with all metrics computed
            dimension_rows = sum([el for el in all_Ks[:-1]])
            dimension_cols = sum([el for el in all_Ks[1:]])

            # Dataframes Creation
            matrix = np.zeros([dimension_rows, dimension_cols])
            Columns_DF = list(itertools.chain(*[[f'K_{kj}_{j}' for j in np.arange(0, kj)] for kj in np.arange(min(all_Ks) + 1, max(all_Ks) + 1)]))
            row_DF = list(itertools.chain(*[[f'K_{kj}_{j}' for j in np.arange(0, kj)] for kj in np.arange(min(all_Ks), max(all_Ks))]))
            Columns_DF_NMI = [f'K_{k}' for k in np.arange(min(all_Ks) + 1, max(all_Ks) + 1)]
            DICE_Similarity_matrix = pd.DataFrame(deepcopy(matrix), columns=Columns_DF, index=row_DF)
            IOU_Similarity_matrix = pd.DataFrame(deepcopy(matrix), columns=Columns_DF, index=row_DF)

            # NMI Matrix computer
            matrix_NMI = np.zeros([dimension_rows, len(Columns_DF_NMI)])
            NMI_matrix = pd.DataFrame(deepcopy(matrix_NMI), columns=Columns_DF_NMI, index=row_DF)
            Adj_NMI_matrix = pd.DataFrame(deepcopy(matrix_NMI), columns=Columns_DF_NMI, index=row_DF)

            # Setting the firts K where to start (i):
            k_rows = np.arange(min(all_Ks), max(all_Ks))
            k_cols = np.arange(min(all_Ks) + 1, max(all_Ks) + 1)

            # Saving the number of datapoint for each Cluster
            Columns_DF_nums = list(itertools.chain(*[[f'K_{kj}_{j}' for j in np.arange(0, kj)] for kj in np.arange(min(all_Ks), max(all_Ks) + 1)]))
            row_DF_ks = [f'K_{kj}' for kj in np.arange(min(all_Ks), max(all_Ks)+1)]
            matrix_nums = np.zeros([row_DF_ks.__len__(), Columns_DF_nums.__len__()])
            Clusters_Dimension_Matrix = pd.DataFrame(deepcopy(matrix_nums), columns=Columns_DF_nums, index=row_DF_ks, dtype=np.int32)
            for row in Clusters_Dimension_Matrix.iterrows():
                Labels_k = self.labels_for_each_K[row[0]]
                l_k, counts_by_label = np.unique(Labels_k['clusters_labels'], return_counts=True)
                for label_k, count in zip(l_k, counts_by_label):
                    Clusters_Dimension_Matrix.loc[row[0], f'{row[0]}_{label_k}'] = count

                print(row)
            for ki, kj in zip(k_rows, k_cols):
                # ------------- ROW ---------------
                # Select the configuration of cluster with K = i:
                lab_ki = self.labels_for_each_K['K_{0}'.format(ki)]
                # Create a vector for each cluster
                l_ki, counts_by_label = np.unique(lab_ki['clusters_labels'], return_counts=True)

                # ------------- COL ---------------
                # Select the configuration of cluster with K = j:
                lab_kj = self.labels_for_each_K['K_{0}'.format(kj)]
                # Create a vector for each cluster
                l_kj = np.unique(lab_kj['clusters_labels'])
                for l_ki_i in l_ki:
                    # Select the SUB- DATAFRAME for ki and label i
                    lab_i_i = lab_ki[lab_ki['clusters_labels'] == l_ki_i]
                    #
                    for l_kj_j in l_kj:
                        # Select the SUB- DATAFRAME for kj and label j
                        lab_j_j = lab_kj[lab_kj['clusters_labels'] == l_kj_j]
                        # COMPUTE DICE COEFFICIENT AND IOU:
                        IOU_ij, DICE_ij = IOU_DICE(lab_i_i=lab_i_i, lab_j_j=lab_j_j)

                        # Locate the results:
                        DICE_Similarity_matrix.loc[f'K_{ki}_{l_ki_i}', f'K_{kj}_{l_kj_j}'] = DICE_ij
                        IOU_Similarity_matrix.loc[f'K_{ki}_{l_ki_i}', f'K_{kj}_{l_kj_j}'] = IOU_ij
                    # Adjusted Mutual Information Score
                    lab_i_i = lab_i_i.drop_duplicates(subset='indexes')
                    int_df = pd.merge(lab_i_i, lab_kj, how='inner', on=['indexes']).drop_duplicates(
                        subset=['indexes']
                    )
                    NMI_i_kj = NMI(labels_true=lab_i_i['patient ID'], labels_pred=int_df['clusters_labels_y'])
                    Adj_NMI_i_kj = ADJ_NMI(labels_true=lab_i_i['patient ID'], labels_pred=int_df['clusters_labels_y'])
                    NMI_matrix.loc[f'K_{ki}_{l_ki_i}', f'K_{kj}'] = NMI_i_kj
                    Adj_NMI_matrix.loc[f'K_{ki}_{l_ki_i}', f'K_{kj}'] = Adj_NMI_i_kj

            return DICE_Similarity_matrix, IOU_Similarity_matrix, NMI_matrix, Adj_NMI_matrix, Clusters_Dimension_Matrix
        except Exception as e:
            print(e)

def IOU_DICE(lab_i_i: pd.DataFrame, lab_j_j: pd.DataFrame):

    # Intesection between two Dataframe
    int_df = pd.merge(lab_i_i, lab_j_j, how='inner', on=['indexes'])
    card_int = int_df.shape[0]
    # Union over two dataframes
    union_df = pd.merge(lab_i_i, lab_j_j, how='outer', on=['indexes'])
    card_union = union_df.shape[0]
    # Card over the two subsets
    cards_i_j = lab_i_i.shape[0] + lab_j_j.shape[0]
    # DICE
    DICE_coeff_i_j = 2 * card_int / cards_i_j
    # IOU
    IOU_i_j = card_int / card_union

    return IOU_i_j, DICE_coeff_i_j,

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
    try:
        assert np.unique(labels_clusters).shape[0] > 1
        # 1) Silhouette_Score:
        Si_score = silhouette_score(Z_latent_samples, labels_clusters)
        # 2) Calinski_Harabasz_score:
        CH_score = calinski_harabasz_score(Z_latent_samples, labels_clusters)
        # 3) davies_bouldin_score:
        DB_score = davies_bouldin_score(Z_latent_samples, labels_clusters)
    except:
        Si_score = 0
        CH_score = 0
        DB_score = 0
        print('Exception launched from metric\'s evaluation task.')
    return {'avg_Si_score': Si_score, 'Calinski-Harabasz score': CH_score, 'Davies-Douldin score': DB_score}


def kmeans(model, dataloader, opt):
    """
    K-means algorithm trained on samples represented Autoencoder latent space.
    Article: MacQueen, J. (1967). Classification and analysis of multivariate observations. In 5th Berkeley Symp. Math. Statist. Probability (pp. 281-297).
    link: https://www.cs.cmu.edu/~bhiksha/courses/mlsp.fall2010/class14/macqueen.pdf
    :param model <BaseModel>: child class of the baseModel class.
    :param data: the same dataset used to pretrain the Autoencoder.
    :param opt (Option class): stores all the experiment flags; needs to be a subclass of BaseOptions
    :return: km (KMeans): returns the kmean algorithm trained on samples from dataloader represented in latent space.
    """
    print('INFO : ---> Initializing cluster centers with k-means.')
    km = KMeans(n_clusters=opt.num_clusters, n_init=100)
    output_array = None
    x_out = None
    print('INFO : ---> Encoding Data on course...')
    output = model.compute_encoded(dataloader=dataloader)
    z_encoded_tot = output['z_latent']
    x_out = output['x_out']

    # Fit k-means algorithm on concatenated samples and predict labels
    print("INFO: ---> Kmeans fitting on course...")
    time_kmeans_0 = time.time()
    prediction = km.fit_predict(z_encoded_tot)
    time_kmeans_f = time.time()
    print("INFO: ---> Kmeans fitted on data \n Time needed for fitting", (time_kmeans_f - time_kmeans_0) / 60, '( min. )')
    return x_out, km, prediction


def weighted_var(values: np.array, weights: np.array) -> np.ndarray:
    """
    Return the weighted average and traditional variance of values.
    values, weights -- Numpy ndarrays with the same shape.
    """
    mean = np.mean(values)
    # Fast and numerically precise:
    variance_w = np.average((values - mean) ** 2, weights=weights)
    return variance_w


def weighted_cov(P1: float, w_1: list, P2: float, w_2) -> np.ndarray:
    mean_1 = np.average(P1, weights=w_1)
    mean_2 = np.average(P2, weights=w_2)


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


def compute_probabilities_variables(
        labels: np.ndarray, probability: np.ndarray, ids: np.ndarray, id_dict: Dict[Any, int], threshold: float = 90
) -> Tuple[float, float, float, np.ndarray, List[int]]:
    """
        This function computes several values related to the input variables 'labels', 'probability', 'ids', and 'id_dict'.
        It also takes an optional input 'threshold' with a default value of 90.
        The computed values include the mean maximum probability, the mean number of prototypes,
        and a mutual information criterion.

        Parameters:
        labels (array-like) : The labels assigned to each element in the input 'ids'
        probability (array-like) : The probability of each label being assigned to each element in the input 'ids'
        ids (array-like) : A unique identifier for each element being labeled
        id_dict (dict) : A dictionary that maps each element in the input 'ids' to a corresponding integer value
        threshold (float) : A threshold value for the maximum probability, used to filter the input data.

        Returns:
        mean_max (float) : The mean maximum probability of the input data
        mean_n_prototypes (float) : The mean number of prototypes for each cluster
        Mutual_information_criterion (float) : The Mutual information criterion calculated from the input data
        indices_th (array-like) : Indices of the elements in the input 'ids' that pass the threshold
        prototypes_for_cluster (list) : A list containing the number of prototypes for each cluster
    """
    # Search for the strings associated with each patient:
    patients = np.unique(ids)
    weights_k = np.bincount(labels)
    weights_k = weights_k[weights_k != 0]
    # Take the assignment probabilities of the labels, and calculate the maximum value:
    real_K = np.unique(labels)
    # Take the maximum probability values, and calculate the mean value:
    saved = probability.max(1)
    # Insert a confidence threshold.
    th = threshold / 100
    # Take the maximum probability values, and calculate the mean value:
    max_values = [el.item() for el in saved]
    mean_max_probabilities = np.mean(max_values)
    # Select both the lebal and the indices according to the threshold.
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

    return mean_max, mean_n_prototypes, Mutual_information_criterion, indices_th, prototypes_for_cluster


def compute_GINI(list_distribution: List[np.ndarray]) -> Tuple[List[float], float]:
    """
    This function computes the Gini index for each array in the input list and returns a tuple containing
    a list of Gini indices and the cumulative Gini index.

    Parameters:
    list_distribution (list): A list of numpy arrays

    Returns:
    gini_index_for_t (list): A list of Gini indices for each array in the input list
    cumulative_gini (float): The cumulative Gini index calculated by summing the individual Gini indices
    """
    gini_index_for_t = [gini(element[np.nonzero(element)].astype(np.float64)) for element in np.array(list_distribution)]
    cumulative_gini = np.sum(gini_index_for_t)
    return gini_index_for_t, cumulative_gini


def calc_Delta_metrics(matrix: np.ndarray) -> Tuple[Any, Any]:
    """
    This function calculates the delta matrix and the delta sum matrix
    and returns all the matrices

    Parameters:
    matrix (ndarray) : The matrix to use to calculate delta metric from

    Returns:
    D_matrix (ndarray): The delta matrix
    D_sum_matrix (ndarray): The delta sum matrix
    """
    D_matrix = np.array([list((matrix[i + 1, :]) - (matrix[i, :])) for i in range(matrix.shape[0] - 1)])
    D_sum_matrix = D_matrix.sum(1)

    return D_matrix, D_sum_matrix


def TF_Variances_ECF(z_: np.array, labels: np.array, ids: np.array) -> Tuple[Any, Any, Any, Any]:
    """
    Compute the variances of the clusters distribution over the patients.

    Parameters:
        - z_ (np.array): matrix of variables where each row represents a data point and each column represents a variable.
        - labels (np.array): vector of labels, where each element corresponds to a data point.
        - ids (np.array): vector of ids, where each element corresponds to a data point.

    Returns:
        - Var_SF (list): list of variances of the cluster distribution over the patients.
        - Var_SF_weighted (list): list of weighted variances of the cluster distribution over the patients.
        - list_Number_of_element_over_t (list): list of the number of elements of each patient over the clusters.
    """
    # Get unique patients IDs
    global log_Var_w
    patients = np.unique(ids)
    N_patients = len(patients)
    # Count the number of elements in each cluster
    count_by_cluster = np.bincount(labels)
    # Variances: rispetto al numero di slices di t (t_t) e al numero di slices di k (t_k).
    Var_SF_w = list()
    Var_log_SF_w = list()
    Var_SF_100_w = list()
    list_Number_of_element_over_t = list()
    # Compute a new list that contains z_ samples and labels for each patient.
    X_for_patient = [z_[[patient == id for id in ids]] for patient in patients]
    labels_for_patient = [labels[[patient == id for id in ids]] for patient in patients]
    # Iteration over patients:
    for t in range(N_patients):
        # Dataset for patient ( Tensor of slices for patient ID )
        X_ = X_for_patient[t]
        # Distribuzioni del paziente nei clusters:
        list_Number_of_element = list()
        list_SF = list()
        list_log_SF = list()
        list_SF_100 = list()
        for k in np.unique(labels):
            # Indexing to find images related to a certain cluster:
            select_t = [label.item() == k for label in labels_for_patient[t]]
            if not len(X_[select_t]) == 0 or True:
                # Frequency between the number of slices of the patient in cluster k and the total number of slices of the patient t:
                sf_t_t = len(X_[select_t]) / labels_for_patient[t].shape[0]
                # Save all values in lists, scalable later with tensors or arrays:
                try:
                    list_SF.append(sf_t_t)

                    list_SF_100.append(sf_t_t*100)
                    list_Number_of_element.append(len(X_[select_t]))
                    assert sf_t_t != 0
                    list_log_SF.append(sf_t_t * 100 * math.log((100 * sf_t_t)))
                except:
                    pass
        # Normal and weighted variance:
        Var_w = weighted_var(np.array(list_SF), np.array(list_Number_of_element))
        log_Var_w = weighted_var(np.array(list_log_SF), np.array(np.array(list_Number_of_element)[np.nonzero
                                                                 (list_Number_of_element)]))
        Var_w_100 = weighted_var(np.array(list_SF_100), np.array(list_Number_of_element))
        # Varianza normale e pesata:
        Var_SF_w.append(Var_w)
        Var_log_SF_w.append(log_Var_w)
        Var_SF_100_w.append(Var_w_100)
        # Distribuzione di slices paziente over clusters:
        list_Number_of_element_over_t.append(list_Number_of_element)

    return Var_log_SF_w, Var_SF_w, Var_SF_100_w, list_Number_of_element_over_t


def target_distribution(batch):
    """
        Compute the target distribution p_ij, given the q_ij (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [q_ij size, number of clusters] Tensor of dtype float
        :return: [q_ij size, number of clusters] Tensor of dtype float"""
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
