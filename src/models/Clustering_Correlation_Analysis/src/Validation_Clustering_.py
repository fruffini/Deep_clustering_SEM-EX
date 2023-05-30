# **SEM-EX CORRELATION ANALYSIS**
from datetime import datetime

import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from numpy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, recall_score
from sklearn.utils import resample


import util_general
from src.models.Clustering_Correlation_Analysis.src.util import util_preprocessing_data
from src.models.Clustering_Correlation_Analysis.src.util.util_models import _train_classifier, _import_model
from src.models.Clustering_Correlation_Analysis.src.util.util_preprocessing_data import Result_Matrix, _Mode_Preprocessing_Selector, _Clustering_Labels_Data_processing_By_K
from util_path import get_next_run_id_local
from pandas.plotting import table
# %% DIRECTORY SAVING / Experiments Enumeration:


# Directory Baseline
data_dir = r'C:\Users\Ruffi\Desktop\Deep_clustering_SEM-EX\src\models\Clustering_Correlation_Analysis\data'

# Data BASELINE Experiment:
file_features = 'RETRO_PR_Dataset_Clinical.xlsx'
Claro_features_PRO_RETRO = os.path.join(data_dir, file_features)

# 1) BASELINE: Clinical features selected from CLARO prospettico:  { features + labels }                                         |||   Labels-Y       |||
#________________________________________________________________________________________________________________________________|||__________________|||
#   ID_patient   |    age    |       sex       |      stage      |  nT  |  nN  |  nM  |  1° Treatment  |  Adiuvante | Chirurgia  |||   OS             |||
#  187 patients  |   float   |   categorical   |   categorical   |  int | int  | int  |   categorical  |    int     |   int      |||   Boolean        |||
#________________________________________________________________________________________________________________________________|||__________________|||

# Data CLUSTERING Experiment:
Clustering_dir = 'DC_Results'
DC_results_dir = os.path.join(data_dir, Clustering_dir)


# Reports Path Validation
reports_dir = r'C:\Users\Ruffi\Desktop\Deep_clustering_SEM-EX\src\models\Clustering_Correlation_Analysis\reports'
#%%

# Experiments ID:
now = datetime.now()
date_time = now.strftime("%Y_%m_%d")
log_path = os.path.join(reports_dir, 'log_run')
run_id = get_next_run_id_local(log_path, 'MY')  # GET run id
run_name = "{0:05d}_{1}".format(run_id, date_time)
# Directory Log Name
log_dir_exp = os.path.join(log_path, run_name)
util_general.mkdir(log_dir_exp)
Baseline_reports_dir = os.path.join(log_dir_exp, 'Baseline_Results')
util_general.mkdir(Baseline_reports_dir)

# Data Loading
Clinical_Claro_features = pd.read_excel(Claro_features_PRO_RETRO, index_col=1).drop(columns=['Unnamed: 0'])

# Common Parameters ( TODO pass to yaml file )
fillnan = 0
random_state = 0
n_splits = 10
# Experiments Parameters
Models_ = ['RF', 'AdaBoost', 'SVM', 'DT', 'MLP', 'TabNet']
Metrics_Eval_ = ['Accuracy', 'F-score', 'Recall', 'AUC']

# Bootstrap Parameters:

n_iterations = 20
size_perc = 0.80

# Modality Selection Between ( Identity, Normalize, Standard)
Mode_selection = 'Identity'

Eval_method = 'k-fold'

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.1) Labels are the values we want to predict:
Y = Clinical_Claro_features.pop('label OS').astype(int)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.2) Features for training:
X_raw = Clinical_Claro_features.copy(deep=True)
# Features that are too much related to labels:
X_raw = X_raw.drop(columns=['Stato'])

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.1) Fill-Nan Operations:
X_raw['cT'] = pd.to_numeric( X_raw['cT']).fillna(fillnan)
X_raw['cN'] = pd.to_numeric( X_raw['cN']).fillna(fillnan)
X_raw['cM'] = pd.to_numeric( X_raw['cM']).fillna(fillnan)


# 3) Obtain the One Hot encoding of all the Data
X = pd.get_dummies(X_raw)
# 4) StratifiedKFold
kf_stratified = StratifiedKFold(n_splits=n_splits)

# %%
# MODEL RESULTS: Creation of a dataframe where store the Results
Results_Matrix_Baseline_Mean = Result_Matrix(Rows=Models_, Columns=Metrics_Eval_)

# %%
# BASELINE
# K-fold Stratified:
Final_Model_Results_By_Fold = {Model_name : None for Model_name in Models_}
for i, Model_name in enumerate(Models_):

    # Importing The Model Name
    Classifier = _import_model(Model_name)
    Rows_Folds = ['Fold_{}'.format(str(Fold_id)) for Fold_id in range(n_splits)]
    Model_Results_By_Fold = Result_Matrix(Rows=Rows_Folds, Columns=Metrics_Eval_)
    # AUC
    mean_fpr = np.linspace(0, 1, 100)
    tp_rates = []
    aucs = []
    # Metrics Eval
    accuracies = []
    F1_scores = []
    Recall = []

    # We spilt each FOLD by index
    for (train, test), Id_Fold in zip(kf_stratified.split(X, Y), Rows_Folds):

        # Separate Training And Test - Features/Labels
        # Fill-NaN Values
        X_train = X.iloc[train].fillna(fillnan)
        X_test = X.iloc[test].fillna(fillnan)
        y_train = Y.iloc[train]
        y_test = Y.iloc[test]

        # normalization
        X_train, max_, min_ = _Mode_Preprocessing_Selector(Mode_selection)(X_train)
        X_test, _, _ = _Mode_Preprocessing_Selector(Mode_selection)(X_test)

        # Train the model
        probs_, y_pred = _train_classifier(Classifier, X_train, y_train, X_test)
        # Accuracy Score
        Model_Results_By_Fold.loc[Id_Fold, Metrics_Eval_[0]] = np.round(accuracy_score(y_test, y_pred), 2)
        # F1-score
        Model_Results_By_Fold.loc[Id_Fold, Metrics_Eval_[1]] = np.round(f1_score(y_test, y_pred),2 )
        # Recall
        Model_Results_By_Fold.loc[Id_Fold, Metrics_Eval_[2]] = np.round(recall_score(y_test, y_pred), 2)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probs_[:, 1])
        tp_rates.append(interp(mean_fpr, fpr, tpr))
        tp_rates[-1][0] = 0.0
        # AUC
        roc_auc = auc(fpr, tpr)
        Model_Results_By_Fold.loc[Id_Fold, Metrics_Eval_[3]] = np.round(roc_auc, 2)
    # Storing NAd Savings Results
    Final_Model_Results_By_Fold[Model_name] = Model_Results_By_Fold
    Baseline_reports_dir
    # Save every single Model
    Model_Table_file = f'Table_{Model_name}_results_FOLDs.tiff'
    plt.figure(figsize=[15, 10])
    ax = plt.subplot()  # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    ax.table(rowLabels=Model_Results_By_Fold.axes[0].array, cellText=Model_Results_By_Fold.values, colLabels=Model_Results_By_Fold.columns,
        loc='center',
        cellLoc='center')  # where df is your data frame

    plt.savefig(os.path.join(Baseline_reports_dir,Model_Table_file))
    plt.show()
    # Adding Up to The mean/std matrix
    Results_Matrix_Baseline_Mean.loc[Model_name, Metrics_Eval_[0]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[0]]), 2) } +/- {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[0]]), 2) }'
    Results_Matrix_Baseline_Mean.loc[Model_name, Metrics_Eval_[1]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[1]]), 2)} +/-  {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[1]]), 2)}'
    Results_Matrix_Baseline_Mean.loc[Model_name, Metrics_Eval_[2]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[0]]), 2) } +/- {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[2]]), 2) }'
    Results_Matrix_Baseline_Mean.loc[Model_name, Metrics_Eval_[3]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[0]]), 2) } +/- {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[3]]), 2) }'

Model_Table_file_all = f'Table_ALL_results_FOLDs.tiff'
ax = plt.subplot()  # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

ax.table(
    rowLabels=Results_Matrix_Baseline_Mean.axes[0].array, cellText=Results_Matrix_Baseline_Mean.values, colLabels=Results_Matrix_Baseline_Mean.columns,
    loc='center',
    cellLoc='center'
    )  # where df is your data frame

plt.savefig(os.path.join(Baseline_reports_dir, Model_Table_file_all))
plt.show()

#%%
# EXPERIMENTAL

file_listed = os.listdir(DC_results_dir)
# We load each configuration for analysis in a dict
print(file_listed)
# Features for training
X_clustering= Clinical_Claro_features.copy(deep=True)
# Features that are too much related to labels:
X_clustering = X_clustering.drop(columns=['Stato'])

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.1) Fill-Nan Operations:
X_clustering['cT'] = pd.to_numeric( X_clustering['cT']).fillna(fillnan)
X_clustering['cN'] = pd.to_numeric( X_clustering['cN']).fillna(fillnan)
X_clustering['cM'] = pd.to_numeric( X_clustering['cM']).fillna(fillnan)

Dataframes_Dict_Ks = {'K_{}'.format(file.split('_')[-2]) : pd.read_excel(os.path.join(DC_results_dir, file)) for file in file_listed}

# Features Mode :

for K_Clustering in Dataframes_Dict_Ks.keys():
    # K Clustering:
    K_df = Dataframes_Dict_Ks[K_Clustering].drop(columns=['Unnamed: 0', 'img ID', 'indexes'])
    Matrices_options = _Clustering_Labels_Data_processing_By_K(K_df=K_df, K_Selected=K_Clustering, )
    # The final Dataset composed of all the informations:
    for Modality, New_features_Clustering in Matrices_options.items():

        # Merging Between The clinical features + Deep Features
        X_clustering = pd.merge(X_clustering, New_features_Clustering, how='inner', on='ID paziente' )

        # Obtain the One Hot encoding of all the Data
        X = pd.get_dummies(X_clustering)

        # MODEL RESULTS: Creation of a dataframe where store the Results

        Experimental_reports_dir = os.path.join(log_dir_exp, '{}_{}_Results'.format(K_Clustering, Modality ))
        util_general.mkdir(Experimental_reports_dir)

        Results_Matrix_Experimental_Mean = Result_Matrix(Rows=Models_, Columns=Metrics_Eval_)

        # BASELINE
        # K-fold Stratified:
        Final_Model_Results_By_Fold = {Model_name : None for Model_name in Models_}
        for i, Model_name in enumerate(Models_):

            # Importing The Model Name
            Classifier = _import_model(Model_name)
            Rows_Folds = ['Fold_{}'.format(str(Fold_id)) for Fold_id in range(n_splits)]
            Model_Results_By_Fold = Result_Matrix(Rows=Rows_Folds, Columns=Metrics_Eval_)
            # AUC
            mean_fpr = np.linspace(0, 1, 100)
            tp_rates = []
            aucs = []
            # Metrics Eval
            accuracies = []
            F1_scores = []
            Recall = []

            # We spilt each FOLD by index
            for (train, test), Id_Fold in zip(kf_stratified.split(X, Y), Rows_Folds):

                # Separate Training And Test - Features/Labels
                # Fill-NaN Values
                X_train = X.iloc[train].fillna(fillnan)
                X_test = X.iloc[test].fillna(fillnan)
                y_train = Y.iloc[train]
                y_test = Y.iloc[test]

                # normalization
                X_train, max_, min_ = _Mode_Preprocessing_Selector(Mode_selection)(X_train)
                X_test, _, _ = _Mode_Preprocessing_Selector(Mode_selection)(X_test)

                # Train the model
                probs_, y_pred = _train_classifier(Classifier, X_train, y_train, X_test)
                # Accuracy Score
                Model_Results_By_Fold.loc[Id_Fold, Metrics_Eval_[0]] = np.round(accuracy_score(y_test, y_pred), 2)
                # F1-score
                Model_Results_By_Fold.loc[Id_Fold, Metrics_Eval_[1]] = np.round(f1_score(y_test, y_pred),2 )
                # Recall
                Model_Results_By_Fold.loc[Id_Fold, Metrics_Eval_[2]] = np.round(recall_score(y_test, y_pred), 2)
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y_test, probs_[:, 1])
                tp_rates.append(interp(mean_fpr, fpr, tpr))
                tp_rates[-1][0] = 0.0
                # AUC
                roc_auc = auc(fpr, tpr)
                Model_Results_By_Fold.loc[Id_Fold, Metrics_Eval_[3]] = np.round(roc_auc, 2)
            # Storing NAd Savings Results
            Final_Model_Results_By_Fold[Model_name] = Model_Results_By_Fold
            Baseline_reports_dir
            # Save every single Model
            Model_Table_file = f'Table_{Model_name}_results_FOLDs.tiff'
            plt.figure(figsize=[15, 10])
            ax = plt.subplot()  # no visible frame
            ax.xaxis.set_visible(False)  # hide the x axis
            ax.yaxis.set_visible(False)  # hide the y axis

            ax.table(rowLabels=Model_Results_By_Fold.axes[0].array, cellText=Model_Results_By_Fold.values, colLabels=Model_Results_By_Fold.columns,
                loc='center',
                cellLoc='center')  # where df is your data frame

            plt.savefig(os.path.join(Experimental_reports_dir ,Model_Table_file))
            plt.show()
            # Adding Up to The mean/std matrix
            Results_Matrix_Baseline_Mean.loc[Model_name, Metrics_Eval_[0]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[0]]), 2) } +/- {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[0]]), 2) }'
            Results_Matrix_Baseline_Mean.loc[Model_name, Metrics_Eval_[1]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[1]]), 2)} +/-  {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[1]]), 2)}'
            Results_Matrix_Baseline_Mean.loc[Model_name, Metrics_Eval_[2]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[0]]), 2) } +/- {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[2]]), 2) }'
            Results_Matrix_Baseline_Mean.loc[Model_name, Metrics_Eval_[3]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[0]]), 2) } +/- {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[3]]), 2) }'

        Model_Table_file_all = f'Table_ALL_results_FOLDs.tiff'
        ax = plt.subplot()  # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis

        ax.table(
            rowLabels=Results_Matrix_Baseline_Mean.axes[0].array, cellText=Results_Matrix_Baseline_Mean.values, colLabels=Results_Matrix_Baseline_Mean.columns,
            loc='center',
            cellLoc='center'
            )  # where df is your data frame

        plt.savefig(os.path.join(Experimental_reports_dir, Model_Table_file_all))
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# %%
# Experimental Bootstrap
# MODEL RESULTS: Creation of a dataframe where store the Results
Baseline_Boostrap_reports_dir = os.path.join(log_dir_exp, 'Baseline_Results_Boostrap')
util_general.mkdir(Baseline_Boostrap_reports_dir)
Results_Matrix_Baseline_BS = Result_Matrix(Rows=Models_, Columns=Metrics_Eval_)
Final_Model_Results_By_Fold = {Model_name : None for Model_name in Models_}
for i, Model_name in enumerate(Models_):

    # Importing The Model Name
    Classifier = _import_model(Model_name)
    Rows_Iters = ['iter_{}'.format(str(iter_id)) for iter_id in range(n_iterations)]

    Model_Results_By_Fold = Result_Matrix(Rows=Rows_Iters, Columns=Metrics_Eval_)

    # Lets create Bootstrapping Dataset

    X_bootstrap = X.copy(deep=True)
    X_bootstrap = pd.concat([X_bootstrap.axes[0].to_frame(), X_bootstrap, Y], axis=1)
    X_values = X_bootstrap.values

    n_size = int(len(X) * size_perc)
    for i, iter in zip(range(n_iterations), Rows_Iters):
        # Prepare train & test sets
        train = resample(X_values, n_samples=n_size)  # Sampling with replacement..whichever is not used in training data will be used in test data
        # Indexes unique ID:
        train_index_unique_ID, indexes_ = np.unique(train[:, 0].astype(int), return_index=True)
        train_uniques = train[indexes_]
        maximum_train = train[:, :-1].max()
        minimum_train = train[:, :-1].min()
        # train[:,:-1] = (train[:,:-1] - minimum_train) / (maximum_train - minimum_train)
        # Test Fold:
        indexes_test = [True if x not in list(train_uniques[:, 0]) else False for x in list(X_values[:, 0])]
        test = X_values[indexes_test]
        # picking rest of the data not considered in training sample
        # test[:,:-1] = (test[:,:-1] - minimum_train) / (maximum_train - minimum_train)
        # Fit model
        X_train = train[:, 1:-1]
        Y_train = train[:, -1]
        Y_test = test[:, -1]
        X_test = test[:, 1:-1]
        Classifier.fit(X_train, Y_train)  # model.fit(X_train,y_train) i.e model.fit(train set, train label as it is a classifier)

        # evaluate model
        predictions = Classifier.predict(X_test)  # model.predict(X_test)

        # evaluate model
        Model_Results_By_Fold.loc[iter, Metrics_Eval_[0]] = np.round(accuracy_score(Y_test, predictions), 2)
        # F1-score
        Model_Results_By_Fold.loc[iter, Metrics_Eval_[1]] = np.round(f1_score(Y_test, predictions), 2)
        # Recall
        Model_Results_By_Fold.loc[iter, Metrics_Eval_[2]] = np.round(recall_score(Y_test, predictions), 2)
    Results_Matrix_Baseline_BS.loc[
        Model_name, Metrics_Eval_[0]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[0]]), 2)} +/- {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[0]]), 2)}'
    Results_Matrix_Baseline_BS.loc[
        Model_name, Metrics_Eval_[1]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[1]]), 2)} +/-  {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[1]]), 2)}'
    Results_Matrix_Baseline_BS.loc[
        Model_name, Metrics_Eval_[2]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[0]]), 2)} +/- {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[2]]), 2)}'

Model_Table_file_all = f'Table_ALL_results_FOLDs_BS.tiff'
ax = plt.subplot()  # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.table(
    rowLabels=Results_Matrix_Baseline_BS.axes[0].array,
    cellText=Results_Matrix_Baseline_BS.values,
    colLabels=Results_Matrix_Baseline_BS.columns,
    loc='center',
    cellLoc='center'
)  # where df is your data frame
plt.savefig(os.path.join(Baseline_Boostrap_reports_dir, Model_Table_file_all))
plt.show()


# %%

# Bootstrap
# EXPERIMENTAL
file_listed = os.listdir(DC_results_dir)
# We load each configuration for analysis in a dict
print(file_listed)
# Features for training
X_clustering = Clinical_Claro_features.copy(deep=True)
# Features that are too much related to labels:
X_clustering = X_clustering.drop(columns=['Stato'])
# 2.1) Fill-Nan Operations:
X_clustering['cT'] = pd.to_numeric( X_clustering['cT']).fillna(fillnan)
X_clustering['cN'] = pd.to_numeric( X_clustering['cN']).fillna(fillnan)
X_clustering['cM'] = pd.to_numeric( X_clustering['cM']).fillna(fillnan)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Dataframes_Dict_Ks = {'K_{}'.format(file.split('_')[-2]) : pd.read_excel(os.path.join(DC_results_dir, file)) for file in file_listed}

# Features Mode :

for K_Clustering in Dataframes_Dict_Ks.keys():
    print('K_Clustering', ' : ', K_Clustering, '/n')
    # K Clustering:
    K_df = Dataframes_Dict_Ks[K_Clustering].drop(columns=['Unnamed: 0', 'img ID', 'indexes'])
    Matrices_options = _Clustering_Labels_Data_processing_By_K(K_df=K_df, K_Selected=K_Clustering, )
    # The final Dataset composed of all the informations:
    for Modality, New_features_Clustering in Matrices_options.items():
        # Importing The Model Name
        print('MODALITY', ' : ', Modality, '/n')
        for i, Model_name in enumerate(Models_):
            Classifier = _import_model(Model_name)
            Rows_Iters = ['iter_{}'.format(str(iter_id)) for iter_id in range(n_iterations)]
            Model_Results_By_Fold = Result_Matrix(Rows=Rows_Iters, Columns=Metrics_Eval_)

            # Merging Between The clinical features + Deep Features
            X_clustering = pd.merge(X_clustering, New_features_Clustering, how='inner', on='ID paziente' )

            # Obtain the One Hot encoding of all the Data
            X = pd.get_dummies(X_clustering)

            # MODEL RESULTS: Creation of a dataframe where store the Results

            Experimental_reports_dir = os.path.join(log_dir_exp, 'Experimental-BootStrap')
            util_general.mkdir(Experimental_reports_dir)

            Results_Matrix_Experimental_Mean_BS = Result_Matrix(Rows=Models_, Columns=Metrics_Eval_)

            # Lets run Bootstrap
            X_bootstrap = X.copy(deep=True)
            X_bootstrap = pd.concat([X_bootstrap.axes[0].to_frame(), X_bootstrap, Y], axis=1)
            X_values = X_bootstrap.values

            n_size = int(len(X) * size_perc)
            for i, iter_ in zip(range(n_iterations), Rows_Iters):
                # Prepare train & test sets
                train = resample(X_values, n_samples=n_size)  # Sampling with replacement..whichever is not used in training data will be used in test data
                # Indexes unique ID:
                train_index_unique_ID, indexes_ = np.unique(train[:, 0].astype(int), return_index=True)
                train_uniques = train[indexes_]
                maximum_train = train[:, :-1].max()
                minimum_train = train[:, :-1].min()
                # train[:,:-1] = (train[:,:-1] - minimum_train) / (maximum_train - minimum_train)
                # Test Fold:
                indexes_test = [True if x not in list(train_uniques[:, 0]) else False for x in list(X_values[:, 0])]
                test = X_values[indexes_test]
                # picking rest of the data not considered in training sample
                # test[:,:-1] = (test[:,:-1] - minimum_train) / (maximum_train - minimum_train)
                # Fit model
                X_train = train[:, 1:-1]
                Y_train = train[:, -1]
                Y_test = test[:, -1]
                X_test = test[:, 1:-1]
                Classifier.fit(X_train, Y_train)  # model.fit(X_train,y_train) i.e model.fit(train set, train label as it is a classifier)

                # evaluate model
                predictions = Classifier.predict(X_test)  # model.predict(X_test)

                # evaluate model
                Model_Results_By_Fold.loc[iter_, Metrics_Eval_[0]] = np.round(accuracy_score(Y_test, predictions), 2)
                # F1-score
                Model_Results_By_Fold.loc[iter_, Metrics_Eval_[1]] = np.round(f1_score(Y_test, predictions), 2)
                # Recall
                Model_Results_By_Fold.loc[iter_, Metrics_Eval_[2]] = np.round(recall_score(Y_test, predictions), 2)
            Results_Matrix_Experimental_Mean_BS.loc[
                Model_name, Metrics_Eval_[0]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[0]]), 2)} +/- {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[0]]), 2)}'
            Results_Matrix_Experimental_Mean_BS.loc[
                Model_name, Metrics_Eval_[1]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[1]]), 2)} +/-  {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[1]]), 2)}'
            Results_Matrix_Experimental_Mean_BS.loc[
                Model_name, Metrics_Eval_[2]] = f'{np.round(np.mean(Model_Results_By_Fold[Metrics_Eval_[0]]), 2)} +/- {np.round(np.std(Model_Results_By_Fold[Metrics_Eval_[2]]), 2)}'

        Model_Table_file_all = '{}_{}_Results Table_ALL_results_FOLDs_BS.tiff'.format(K_Clustering, Modality)
        ax = plt.subplot()  # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        ax.table(
            rowLabels=Results_Matrix_Experimental_Mean_BS.axes[0].array,
            cellText=Results_Matrix_Experimental_Mean_BS.values,
            colLabels=Results_Matrix_Experimental_Mean_BS.columns,
            loc='center',
            cellLoc='center'
        )  # where df is your data frame
        plt.savefig(os.path.join(Experimental_reports_dir, Model_Table_file_all))
        plt.show()

