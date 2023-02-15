# **SEM-EX CORRELATION ANALYSIS**
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from scipy.special import softmax
from src.models.Clustering_Correlation_Analysis.src.util import util_preprocessing_data

# %% DIRECTORY SAVING:














# 1) BASELINE: Clinical features selected from CLARO prospettico:  { features + labels }
#   ID_patient   |    age    |       sex       |      stage      |  nT  |  nN  |  nM  |  1° Treatment  |  Adiuvante | Chirurgia  |||   OS
#  186 patients  |   float   |   categorical   |   categorical   |  int | int  | int  |   categorical  |    int     |   int      |||   Boolean


#  Config File import


# Data Import

Claro_features_PRO = r'C:\Users\Ruffi\Desktop\Deep_clustering_SEM-EX\src\models\Clustering_Correlation_Analysis\data\RETRO_PR_Dataset_Clinical.xlsx'
Clinical_Claro_features = pd.read_excel(Claro_features_PRO, index_col=1).drop(columns=['Unnamed: 0'])


# Common Parameters ( TODO pass to yaml file )
fillnan = 0
random_state = 42
n_splits = 10

# Bootstrap Parameters:

n_iterations = 20
size_perc = 0.70


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.1) Labels are the values we want to predict:
Y = Clinical_Claro_features.pop('label OS').astype(int)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.2) Features for training:
X_raw= Clinical_Claro_features.copy(deep=True)
# Features that too much related to labels:
X_raw = X_raw.drop(columns=[ 'Stato'])

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.1) Fill-Nan Operations:
X_raw['cT'] = pd.to_numeric( X_raw['cT']).fillna(fillnan)
X_raw['cN'] = pd.to_numeric( X_raw['cN']).fillna(fillnan)
X_raw['cM'] = pd.to_numeric( X_raw['cM']).fillna(fillnan)


# 3) Obtain the One Hot encoding of all the Data
X = pd.get_dummies(X_raw)
#4) kf StratifiedKFold
kf=StratifiedKFold(n_splits=n_splits)

# %%

# Instantiate model with 1000 decision trees
rf_classifier = RandomForestClassifier(random_state=random_state)

# Plot the ROC curves combined
Accuracies_baseline, rf_classifier_baseline = util_preprocessing_data.plot_training_acc_and_Combined_ROC_CURVE(
    X=X,
    Y=Y,
    classifier=rf_classifier,
    kf=kf,
    Experiment_name='RF, Baseline Clinical features')

print(f"The accuracy of the model is:  { np.mean(list(Accuracies_baseline.values()))*100} +/- {np.std(list(Accuracies_baseline.values()))} %")





# %%
# Bootstrap
#Lets run Bootstrap
X_bootstrap = X.copy(deep=True)
X_bootstrap = pd.concat( [ X_bootstrap.axes[0].to_frame(), X_bootstrap, Y], axis=1)
X_values = X_bootstrap.values

n_size = int(len(X) * size_perc)
Accuracies_bootstrap = dict()
for i in range(n_iterations):

    # Prepare train & test sets
    train = resample(X_values, n_samples = n_size) #Sampling with replacement..whichever is not used in training data will be used in test data
    # Indexes unique ID:
    train_index_unique_ID, indexes_ = np.unique(train[:, 0].astype(int), return_index=True)
    train_uniques = train[indexes_]
    maximum_train = train[:, :-1].max()
    minimum_train = train[:, :-1].min()
    #train[:,:-1] = (train[:,:-1] - minimum_train) / (maximum_train - minimum_train)
    # Test Fold:
    indexes_test = [True if x not in list(train_uniques[:,0]) else False for x in list(X_values[:, 0]) ]
    test = X_values[indexes_test]
    #picking rest of the data not considered in training sample
    # test[:,:-1] = (test[:,:-1] - minimum_train) / (maximum_train - minimum_train)
    # Fit model
    model = RandomForestClassifier(random_state=random_state)
    X_train = train[:,1:-1]
    Y_train = train[:, -1]
    model.fit(X_train, Y_train) #model.fit(X_train,y_train) i.e model.fit(train set, train label as it is a classifier)

    #evaluate model
    predictions = model.predict(test[:,1:-1]) #model.predict(X_test)
    Accuracies_bootstrap[i] = accuracy_score(test[:,-1], predictions) #accuracy_score(y_test, y_pred)
# Printing the Accuracy
print(f"The accuracy of the model is:  { np.mean(list(Accuracies_bootstrap.values()))*100} +/- {np.std(list(Accuracies_bootstrap.values()))} %")


# %%

import os
dir_clustering_configurations = r"C:\Users\Ruffi\Desktop\Deep_clustering_SEM-EX\src\models\Clustering_Correlation_Analysis\data\DC_Results"
file_listed = os.listdir(dir_clustering_configurations)
# We load each configuration for analysis in a dict
print(file_listed)
Dataframes_Dict_ = {'K_{}'.format(file.split('_')[-2]) : pd.read_excel(os.path.join(dir_clustering_configurations, file)) for file in file_listed}


# Selection for the Number of Clusters
Selected = 'K_11'
# Features Mode :
Mode_ = 'counts'
matrix_options = ['boolean', 'softmax', 'counts']
Matrices_ = {option: None for option in matrix_options}


matrix = matrix_options[0]

if Selected in Dataframes_Dict_.keys():
    # Data for the Clusters configuration
    k_ = Dataframes_Dict_[Selected].drop(columns= ['Unnamed: 0', 'img ID', 'indexes'])

    # Dictionary ID -> Cluster Distribution
    Dict_ID_to_labels = k_.groupby('patient ID')['clusters_labels'].apply(list).to_dict()
    Dict_labels_to_ID = k_.groupby('clusters_labels')['patient ID'].apply(list).to_dict()

    # This dict contains all the cluster where the patient has been discovered:
    Dict_ID_to_labels_uniques = {keys: list(np.unique(values)) for keys, values in Dict_ID_to_labels.items()}

    # This dict contains all the patients slices for each cluster:
    Dict_labels_to_uniques_ID = {keys: list(np.unique(values)) for keys, values in Dict_labels_to_ID.items()}
    # ANALyzing all the labels inside the clusters
    # Labels Count
    Dict_ID_to_labels_count= {keys: list(tuple(np.unique(values, return_counts=True))[1]) for keys, values in       Dict_ID_to_labels.items()}
    Dict_labels_to_uniques_count = {keys: list(tuple(np.unique(values, return_counts=True))[1]) for keys, values in Dict_labels_to_ID.items()}
    # Softmax
    Dict_ID_to_labels_softmax= {keys: list(softmax(tuple(np.unique(values, return_counts=True))[1])) for keys, values in Dict_ID_to_labels.items()}
    Dict_labels_to_uniques_softmax = {keys: list(softmax(tuple(np.unique(values, return_counts=True))[1])) for keys, values in Dict_labels_to_ID.items()}
    # Dataframe For Keys and new features
    matrix_len = [len(Dict_ID_to_labels), int(Selected.split('_')[-1])]
    # Separation For each Matrix type:
    for option in Matrices_.keys():
        New_features = pd.DataFrame(np.zeros(matrix_len), index=Dict_ID_to_labels.keys())
        # Options matrices
        for ID, softmax_values in Dict_ID_to_labels_softmax.items():
            counts = Dict_ID_to_labels_uniques[ID]
            if option == 'softmax':
                for i,c in enumerate(counts):
                    New_features.loc[ID, c] = softmax_values[i]
                    New_features.index.name = 'ID paziente'
                    New_features.fillna(0, inplace=True)
                    Matrices_[option] = New_features

            elif option == 'counts':
                for i,c in enumerate(counts):
                    New_features.loc[ID, c] = c
                    New_features.index.name = 'ID paziente'
                    New_features.fillna(0, inplace=True)
                    Matrices_[option] = New_features

            elif option == 'boolean':
                for i,c in enumerate(counts):
                    New_features.loc[ID, c] = 1
                    New_features.index.name = 'ID paziente'
                    New_features.fillna(0, inplace=True)
                    Matrices_[option] = New_features


# Features for training
X_clustering= Clinical_Claro_features.copy(deep=True)
# The final Dataset composed of all the informations:
New_features_Clustering = Matrices_[Mode_]
# Merging Between The clinical features + Deep Features
X_clustering = pd.merge(X_clustering, New_features_Clustering, how='inner', on='ID paziente' )

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1) Features for training:
X_raw= X_clustering.copy(deep=True)
# Features that too much related to labels:
X_raw = X_raw.drop(columns=[ 'Stato'])

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.1) Fill-Nan Operations:
X_raw['cT'] = pd.to_numeric( X_raw['cT']).fillna(fillnan)
X_raw['cN'] = pd.to_numeric( X_raw['cN']).fillna(fillnan)
X_raw['cM'] = pd.to_numeric( X_raw['cM']).fillna(fillnan)


# 3) Obtain the One Hot encoding of all the Data
X = pd.get_dummies(X_raw)
#%% md


# Instantiate model with 1000 decision trees
rf_classifier = RandomForestClassifier(random_state=random_state)


Accuracies_clustering, rf_classifier_clustering= util_preprocessing_data.plot_training_acc_and_Combined_ROC_CURVE(X=X, Y=Y, classifier=rf_classifier, kf=kf, Experiment_name=f'Clustering Mode, {Selected} + {Mode_}')


print(f"The accuracy of the model is:  { np.mean(list(Accuracies_clustering.values()))*100} +/- {np.std(list(Accuracies_clustering.values()))} %")
# Pass these features inside the  for the one hot encoder:

# %%
# Bootstrap
#Lets run Bootstrap
X_bootstrap = X.copy(deep=True)
X_bootstrap = pd.concat( [ X_bootstrap.axes[0].to_frame(), X_bootstrap, Y], axis=1)
X_values = X_bootstrap.values

n_size = int(len(X) * size_perc)
Accuracies_bootstrap = dict()
for i in range(n_iterations):

    # Prepare train & test sets
    train = resample(X_values, n_samples = n_size) #Sampling with replacement..whichever is not used in training data will be used in test data
    # Indexes unique ID:
    train_index_unique_ID, indexes_ = np.unique(train[:, 0].astype(int), return_index=True)
    train_uniques = train[indexes_]
    maximum_train = train[:, :-1].max()
    minimum_train = train[:, :-1].min()
    #train[:,:-1] = (train[:,:-1] - minimum_train) / (maximum_train - minimum_train)
    # Test Fold:
    indexes_test = [True if x not in list(train_uniques[:,0]) else False for x in list(X_values[:, 0]) ]
    test = X_values[indexes_test]
    #picking rest of the data not considered in training sample
    # test[:,:-1] = (test[:,:-1] - minimum_train) / (maximum_train - minimum_train)
    # Fit model
    model = RandomForestClassifier(random_state=random_state)
    X_train = train[:,1:-1]
    Y_train = train[:, -1]
    model.fit(X_train, Y_train) #model.fit(X_train,y_train) i.e model.fit(train set, train label as it is a classifier)

    #evaluate model
    predictions = model.predict(test[:,1:-1]) #model.predict(X_test)
    Accuracies_bootstrap[i] = accuracy_score(test[:,-1], predictions) #accuracy_score(y_test, y_pred)
# Printing the Accuracy
print(f"The accuracy of the model is:  { np.mean(list(Accuracies_bootstrap.values()))*100} +/- {np.std(list(Accuracies_bootstrap.values()))} %")



