import numpy as np
import pandas as pd
from scipy.special import softmax

def _Mode_Preprocessing_Selector(Mode=str):
    if Mode == 'Normalize':
        return Normalize
    elif Mode == 'Standard':
        return Identify  # todo scrivere funzione per la standardizzazione
    elif Mode == 'Identity':
        return Identify
def Result_Matrix(Rows, Columns):
    return pd.DataFrame(data=None, columns=Columns, index=Rows)
def Identify(df):
    return df.copy(), 0, 0




def Normalize(df, min_=None, max_=None):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max() if max_ is None else max_
        min_value = df[feature_name].min() if min_ is None else min_

        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return tuple(result, max_value, min_value)


def _Clustering_Labels_Data_processing_By_K(K_df, K_Selected, matrix_options=['boolean', 'softmax', 'counts']):

    # return the dict with resulting DFs
    Matrices_ = {option: None for option in matrix_options}
    # Data for the Clusters configuration

    # Dictionary ID -> Cluster Distribution
    Dict_ID_to_labels = K_df.groupby('patient ID')['clusters_labels'].apply(list).to_dict()
    Dict_labels_to_ID = K_df.groupby('clusters_labels')['patient ID'].apply(list).to_dict()
    # This dict contains all the cluster where the patient has been discovered:
    Dict_ID_to_labels_uniques = {keys: list(np.unique(values)) for keys, values in Dict_ID_to_labels.items()}
    # This dict contains all the patients slices for each cluster:
    Dict_labels_to_uniques_ID = {keys: list(np.unique(values)) for keys, values in Dict_labels_to_ID.items()}
    # ANALyzing all the labels inside the clusters
    # Labels Count
    Dict_ID_to_labels_count = {keys: list(tuple(np.unique(values, return_counts=True))[1]) for keys, values in Dict_ID_to_labels.items()}
    Dict_labels_to_uniques_count = {keys: list(tuple(np.unique(values, return_counts=True))[1]) for keys, values in Dict_labels_to_ID.items()}
    # Softmax
    Dict_ID_to_labels_softmax = {keys: list(softmax(tuple(np.unique(values, return_counts=True))[1])) for keys, values in Dict_ID_to_labels.items()}
    Dict_labels_to_uniques_softmax = {keys: list(softmax(tuple(np.unique(values, return_counts=True))[1])) for keys, values in Dict_labels_to_ID.items()}
    # Dataframe For Keys and new features
    matrix_len = [len(Dict_ID_to_labels), int(K_Selected.split('_')[-1])]
    # Separation For each Matrix type:
    for option in Matrices_.keys():
        New_features = pd.DataFrame(np.zeros(matrix_len), index=Dict_ID_to_labels.keys())
        # Options matrices
        for ID, softmax_values in Dict_ID_to_labels_softmax.items():
            counts = Dict_ID_to_labels_uniques[ID]
            if option == 'softmax':
                for i, c in enumerate(counts):
                    New_features.loc[ID, c] = softmax_values[i]
                    New_features.index.name = 'ID paziente'
                    New_features.fillna(0, inplace=True)
                    Matrices_[option] = New_features
            elif option == 'counts':
                for i, c in enumerate(counts):
                    New_features.loc[ID, c] = c
                    New_features.index.name = 'ID paziente'
                    New_features.fillna(0, inplace=True)
                    Matrices_[option] = New_features
            elif option == 'boolean':
                for i, c in enumerate(counts):
                    New_features.loc[ID, c] = 1
                    New_features.index.name = 'ID paziente'
                    New_features.fillna(0, inplace=True)
                    Matrices_[option] = New_features
    return Matrices_