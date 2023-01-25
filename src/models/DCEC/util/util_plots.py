from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from util import util_clustering


def plt_probabilities_NMI(file, save_dir):
    data = pd.read_csv(file)
    keys = data.keys()
    K_ = np.array(data["K"])
    matrix_mean_P = [np.mean(list(map(float, i[1:-1].split(', ')))) for i in data["P_for_cluster"]]
    matrix_NMI = np.array(data["Mutual_Information_Score"])
    plt.close()
    file_name = "APN_NMI_over_k"
    fig1, ax = plt.subplots(nrows=1, figsize=(25, 16))
    Mutual_Information_Score = (matrix_NMI - np.min(matrix_NMI)) / (np.max(matrix_NMI) - np.min(matrix_NMI))
    mean_assegnation = (matrix_mean_P - np.min(matrix_mean_P)) / (np.max(matrix_mean_P) - np.min(matrix_mean_P))
    Prod = np.sqrt(Mutual_Information_Score * mean_assegnation)
    data_plot1 = Mutual_Information_Score.copy()
    ax.plot(K_, data_plot1, 'g', label="$NMI_{01}$", linewidth=5, linestyle="dotted")
    data_plot3 = mean_assegnation.copy()
    ax.plot(K_, data_plot3, 'b', label="$APN_{01}$", linewidth=5, linestyle="dotted")
    data_plot2 = Prod.copy()
    ax.plot(K_, data_plot2, 'r', label="$\sqrt{NMI_{01} * APN_{01}}$", linewidth=5, )
    ax.set_xticks(K_)
    ax.set_xticklabels(K_, fontsize=22.0)
    ax.set_ylabel('$\sqrt{NMI_{01} * APN_{01}}$', fontsize=25.0)  # Y label
    ax.set_xlabel('k di inizializzazione', fontsize=25)  # X label
    ax.grid()
    ax.legend(fontsize=28)
    fig1.suptitle(
        'Plot APN, NMI, e metrica combinata $\mathbf{\sqrt{NMI_{01} * APN_{01}}}$', fontsize=28, y=0.94,
        fontweight='bold'
    )
    plt.yticks(fontsize=22)
    fig1.savefig(os.path.join(save_dir, '{}___.png'.format(file_name)))
    plt.close(fig1)


def plt_probabilities_NMI_mean_var(data, save_dir, Ks):
    K_ = np.array(Ks)
    plt.close()
    plt.clf()
    file_name = "Metrics_Combined-APN_NMI_over_EXPs"
    Metrics_mean = data['Metrics_Combined_mean']
    Metrics_var = data['Metrics_Combined_var']
    plt.figure(figsize=(25, 12))
    plt.plot(
        K_, Metrics_mean, 'acqua',
        label="$mean_{t \in T} VAR_wt$ al variare di k", linewidth=6, )
    plt.errorbar(
        x=K_, y=Metrics_mean, yerr=Metrics_var, color='r', linestyle='None', linewidth=3, fmt='o', label='St-Dev'
    )
    plt.xticks(K_, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    plt.xlabel('k di inizializzazione', fontsize=25)
    plt.ylabel('$\sqrt{NMI_{01} * APN_{01}}$', fontsize=25.0, color="k", position=[-1, 0], )
    plt.suptitle(
        'Plot Metric Combined $\mathbf{\sqrt{NMI_{01} * APN_{01}}}$', fontsize=28, y=0.94,
        fontweight='bold'
    )
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.show()
    plt.clf()


def plt_Var_Gini_K_EXPs_mean_var(data, save_dir, Ks):
    K_ = np.array(Ks)
    keys = list(data.keys())

    matrix_gini_mean = np.array(data['Gini_mean'])
    matrix_gini_var = np.array(data['Gini_var'])
    matrix_Var_w_mean = np.array(data['Var_w_mean'])

    matrix_Var_w_var = np.array(data['Var_w_var'])

    N_K = K_.shape[0]
    tabs = ["r", "k"]
    # _________________________________________________________________________________________________________________
    # Plot Varianza

    # _________________________________________________________________________________________________________________
    # Plot Varianza Pesata
    Delta_Var_w_matrix_mean, D_sum_Var_w_matrix_mean = util_clustering.calc_Delta_metrics(
        matrix_Var_w_mean[:, np.newaxis]
    )
    Delta_Var_w_matrix_var, D_sum_Var_w_matrix_var = util_clustering.calc_Delta_metrics(
        matrix_Var_w_var[:, np.newaxis]
    )

    file_name = "Mean_Var_w_over_EXPs"
    plt.clf()
    plt.close()
    plt.figure(1, figsize=(25, 12))
    data_plot = matrix_Var_w_mean.copy()
    plt.plot(
        K_, data_plot, tabs[0],
        label="$mean_{t \in T} VAR_wt$ al variare di k", linewidth=6, )
    plt.errorbar(
        x=K_, y=data_plot, yerr=matrix_Var_w_mean[:, 0], color='k', linestyle='None', linewidth=3, fmt='o', label='St-Dev'
    )
    plt.xticks(K_, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    plt.xlabel('k di inizializzazione', fontsize=25)
    plt.ylabel("$\mathbf{mean_{t \in T} VAR_wt}$", color="k", position=[-1, 0], fontsize=25, )
    plt.suptitle(
        'Plot dell\'andamento $\mathbf{mean_{t \in T} VAR_wt}$ al variare di k ', fontsize=28, y=0.94,
        fontweight='bold'
    )
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.show()
    plt.clf()

    # derivata discreta tra K+1 e K
    file_name = "Delta_Var_w_over_EXPs"
    plt.figure(2, figsize=(25, 15))
    plt.clf()
    plt.grid()
    for i in range(Delta_Var_w_matrix_mean.shape[0]):
        plt.bar(K_[i + 1], D_sum_Var_w_matrix_mean[i], linewidth=3, label=str(K_[i + 1]) + "-" + str(K_[i]))
        plt.errorbar(K_[i + 1], D_sum_Var_w_matrix_mean[i], yerr=D_sum_Var_w_matrix_var[i], fmt="o", color="k")
    plt.ylabel("$\mathbf{\Delta_{k} \sum_{t \in T}VAR_wt} $", fontsize=28, color='k')
    plt.xlabel('k+1  -  k', fontsize=25, color='k', )
    # plt.xticks(K_[1:], [str(K_[i + 1]) + "-" + str(K_[i]) for i in range(matrix_delta_var.shape[0])])
    plt.xticks(K_[1:], [str(K_[i + 1]) + "-" + str(K_[i]) for i in range(D_sum_Var_w_matrix_mean.shape[0])], fontsize=22)
    plt.yticks(fontsize=22)
    plt.suptitle(
        'Plot dell\'andamento $\mathbf{\Delta_k \sum_{t \in T}VAR_wt}$ tra k+1 e k, con $\mathbf{k \in [4, 23]}$',
        fontsize=28, y=0.94, fontweight='bold'
    )
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.show()

    # _________________________________________________________________________________________________________________
    # Plot GINI

    Delta_Gini_mean, Delta_Gini_Sum_mean = util_clustering.calc_Delta_metrics(matrix_gini_mean[:, np.newaxis])
    Delta_Gini_var, Delta_Gini_Sum_var = util_clustering.calc_Delta_metrics(matrix_gini_var[:, np.newaxis])

    file_name = "Gini_mean_t_over_EXPs"
    data_plot = matrix_gini_mean.copy()
    plt.figure(3, figsize=(25, 12))
    plt.clf()

    plt.errorbar(
        x=K_, y=data_plot, yerr=matrix_gini_mean[:, 0], color='r', linestyle='None', linewidth=3, fmt='o', label='St-Dev'
    )
    plt.plot(
        K_, data_plot, tabs[0],
        label="$\mathbf{mean_{t \in T} GINI_t}$ al variare di k", linewidth=5, color="b"
    )
    plt.xticks(K_, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    plt.xlabel('k di inizializzazione', fontsize=25)
    plt.ylabel("$\mathbf{mean_{t \in T} GINI_t}$", color="k", position=[-1, 0], fontsize=25)
    plt.suptitle(
        'Plot dell\'andamento $\mathbf{mean_{t \in T} GINI_t}$ al variare di k ', fontweight='bold',
        fontsize=28, y=0.94
    )
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.close()
    plt.clf()

    # derivata discreta tra K+1 e K

    file_name = "Gini_derivative_over_Exps"
    plt.figure(4, figsize=(25, 15))
    plt.clf()
    plt.grid()
    for i in range(Delta_Var_w_matrix_mean.shape[0]):
        plt.bar(K_[i + 1], Delta_Gini_Sum_mean[i], linewidth=3, label=str(K_[i + 1]) + "-" + str(K_[i]))
        plt.errorbar(K_[i + 1], Delta_Gini_Sum_mean[i], yerr=Delta_Gini_Sum_var[i], fmt="o", color="k")
    plt.ylabel("$ \mathbf{\Delta_{k}\sum_{t \in T} GINI_t }$", fontsize=25, color='k')
    plt.xlabel('k+1  -  k', fontsize=25, color='k', )
    # plt.xticks(K_[1:], [str(K_[i + 1]) + "-" + str(K_[i]) for i in range(matrix_delta_var.shape[0])])
    plt.xticks(K_[1:], [str(K_[i + 1]) + "-" + str(K_[i]) for i in range(Delta_Var_w_matrix_mean.shape[0])], fontsize=20)
    plt.yticks(fontsize=22)
    plt.suptitle(
        'Plot dell\'andamento $\mathbf{\Delta_k  \sum_{t \in T}GINI_t}$ tra k+1 e k, con $\mathbf{ k \in [4, 23]}$',
        fontsize=28, y=0.94, fontweight='bold'
    )
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.show()
    plt.clf()


def plt_Var_Gini_K(file, save_dir):
    data = pd.read_csv(file)
    keys = data.keys()
    K_ = np.array(data["K"])
    matrix_gini = np.array([tuple(map(float, i[1:-1].split(', '))) for i in data["Gini"]])
    matrix_Var_w = np.array([tuple(map(float, i[1:-1].split(', '))) for i in data["Var_w"]])
    N_K = K_.shape[0]
    tabs = ["r", "k"]
    # _________________________________________________________________________________________________________________
    # Plot Varianza

    # _________________________________________________________________________________________________________________
    # Plot Varianza Pesata
    Delta_Var_w_matrix, D_sum_Var_w_matrix = util_clustering.calc_Delta_metrics(matrix_Var_w)
    file_name = f"Mean_Var_w_over_k"
    plt.clf()
    plt.close()
    plt.figure(1, figsize=(25, 12))

    data_plot = matrix_Var_w.copy()
    if data_plot.shape[1] > 20:
        print("here")
    else:
        for i in range(data_plot.shape[1]):
            plt.plot(K_, data_plot[:, i], linewidth=3, linestyle="dashdot", label=" digit " + str(i))
            plt.legend(fontsize=20)

    plt.plot(
        K_, np.array([np.mean(data_plot[i, :]) for i in range(N_K)]), tabs[0],
        label="$mean_{t \in T} VAR_wt$ al variare di k", linewidth=5, )
    plt.xticks(K_, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    plt.xlabel('k di inizializzazione', fontsize=25)
    plt.ylabel("$\mathbf{mean_{t \in T} VAR_wt}$", color="k", position=[-1, 0], fontsize=25, )
    plt.suptitle(
        'Plot dell\'andamento $\mathbf{mean_{t \in T} VAR_wt}$ al variare di k ', fontsize=28, y=0.94,
        fontweight='bold'
    )
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.close()
    plt.clf()

    # derivata discreta tra K+1 e K
    file_name = f"Delta_Var_w_over_k",
    plt.figure(2, figsize=(25, 15))
    plt.clf()
    plt.grid()
    for i in range(Delta_Var_w_matrix.shape[0]):
        plt.bar(K_[i + 1], D_sum_Var_w_matrix[i], linewidth=3, label=str(K_[i + 1]) + "-" + str(K_[i]))

    plt.ylabel("$\mathbf{\Delta_{k} \sum_{t \in T}VAR_wt} $", fontsize=28, color='k')
    plt.xlabel('k+1  -  k', fontsize=25, color='k', )
    # plt.xticks(K_[1:], [str(K_[i + 1]) + "-" + str(K_[i]) for i in range(matrix_delta_var.shape[0])])
    plt.xticks(K_[1:], [str(K_[i + 1]) + "-" + str(K_[i]) for i in range(D_sum_Var_w_matrix.shape[0])], fontsize=22)
    plt.yticks(fontsize=22)
    plt.suptitle(
        'Plot dell\'andamento $\mathbf{\Delta_k \sum_{t \in T}VAR_wt}$ tra k+1 e k, con $\mathbf{k \in [4, 23]}$',
        fontsize=28,
        y=0.94,
        fontweight='bold'
    )
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.close()
    # _________________________________________________________________________________________________________________
    # Plot GINI

    Delta_Gini, Delta_Gini_Sum = util_clustering.calc_Delta_metrics(matrix_gini)

    file_name = f"Gini_mean_t_over_k",
    data_plot = matrix_gini.copy()
    plt.figure(3, figsize=(25, 12))
    plt.clf()
    if data_plot.shape[1] > 20:
        print("here")
    else:
        for i in range(data_plot.shape[1]):
            plt.plot(K_, data_plot[:, i], linewidth=3, linestyle="dashdot", label=" digit " + str(i))
            plt.legend(fontsize=20)
    plt.plot(
        K_, np.array([np.mean(data_plot[i, :]) for i in range(N_K)]), tabs[0],
        label="$\mathbf{mean_{t \in T} GINI_t}$ al variare di k", linewidth=5, color="b"
    )
    plt.xticks(K_, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    plt.xlabel('k di inizializzazione', fontsize=25)
    plt.ylabel("$\mathbf{mean_{t \in T} GINI_t}$", color="k", position=[-1, 0], fontsize=25)
    plt.suptitle(
        'Plot dell\'andamento $\mathbf{mean_{t \in T} GINI_t}$ al variare di k ', fontweight='bold',
        fontsize=28, y=0.94
    )
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.close()
    plt.clf()

    # derivata discreta tra K+1 e K
    file_name = "Delta_GINI_over_k"

    plt.figure(4, figsize=(25, 15))
    plt.clf()
    plt.grid()
    for i in range(Delta_Var_w_matrix.shape[0]):
        plt.bar(K_[i + 1], Delta_Gini_Sum[i], linewidth=3, label=str(K_[i + 1]) + "-" + str(K_[i]))

    plt.ylabel("$ \mathbf{\Delta_{k}\sum_{t \in T} GINI_t }$", fontsize=25, color='k')
    plt.xlabel('k+1  -  k', fontsize=25, color='k', )
    # plt.xticks(K_[1:], [str(K_[i + 1]) + "-" + str(K_[i]) for i in range(matrix_delta_var.shape[0])])
    plt.xticks(K_[1:], [str(K_[i + 1]) + "-" + str(K_[i]) for i in range(D_sum_Var_w_matrix.shape[0])], fontsize=20)
    plt.yticks(fontsize=22)
    plt.suptitle(
        'Plot dell\'andamento $\mathbf{\Delta_k  \sum_{t \in T}GINI_t}$ tra k+1 e k, con $\mathbf{ k \in [4, 23]}$',
        fontsize=28, y=0.94, fontweight='bold'
    )
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.close()
    plt.clf()


def plt_Var_new_metrics(file, save_dir, opt):



    import json
    with open(file, 'r') as file_metrics:
        dict_ = json.load(file_metrics)

    # _________________________________________________________________________________________________________________
    # Plot Var
    # _________________________________________________________________________________________________________________
    key_ = 'VAR_w'
    file_name = f"{key_}_over_k_mean_median"
    plt.clf()
    plt.close()
    plt.figure(1, figsize=(25, 12))
    var_ = dict_[key_]
    K_ = np.arange(opt.k_0, opt.k_fin + 1)
    data_mean_all_t = [np.mean(var_in_k) for var_in_k in var_]
    data_median_all_t = [np.median(var_in_k) for var_in_k in var_]
    plt.plot(K_, data_median_all_t, linewidth=3, color='r', label='Median of Var SF')
    plt.plot(K_, data_mean_all_t, linewidth=3, color='b', label='Mean of Var SF')
    plt.legend(fontsize=25)
    plt.suptitle(
        f'Mean and Median of VAR of patient {key_}', fontsize=25, y=0.91,
        fontweight="bold"
    )
    plt.xticks(K_, fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.close()
    plt.clf()
    # _________________________________________________________________________________________________________________
    # Plot log Var
    # _________________________________________________________________________________________________________________
    key_ = 'log_VAR_w'
    file_name = f"{key_}_over_k_mean_median"
    plt.figure(2, figsize=(25, 12))
    var_ = dict_[key_]
    K_ = np.arange(opt.k_0, opt.k_fin + 1)
    data_mean_all_t = [np.mean(var_in_k) for var_in_k in var_]
    data_median_all_t = [np.median(var_in_k) for var_in_k in var_]
    plt.plot(K_, data_median_all_t, linewidth=3, color='r', label='Median of Var SF')
    plt.plot(K_, data_mean_all_t, linewidth=3, color='b', label='Mean of Var SF')
    plt.legend(fontsize=25)
    plt.suptitle(
        f'Mean and Median of VAR of patient {key_}', fontsize=25, y=0.91,
        fontweight="bold"
    )
    plt.xticks(K_, fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.close()
    plt.clf()
    # _________________________________________________________________________________________________________________
    # Plot log Var
    # _________________________________________________________________________________________________________________
    key_ = 'VAR_w_100'
    file_name = f"{key_}_over_k_mean_median"
    plt.figure(3, figsize=(25, 12))
    var_ = dict_[key_]
    K_ = np.arange(opt.k_0, opt.k_fin + 1)
    data_mean_all_t = [np.mean(var_in_k) for var_in_k in var_]
    data_median_all_t = [np.median(var_in_k) for var_in_k in var_]
    plt.plot(K_, data_median_all_t, linewidth=3, color='r', label='Median of Var SF')
    plt.plot(K_, data_mean_all_t, linewidth=3, color='b', label='Mean of Var SF')
    plt.legend(fontsize=25)
    plt.suptitle(
        f'Mean and Median of VAR of patient {key_}', fontsize=25, y=0.91,
        fontweight="bold"
    )
    plt.xticks(K_, fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.close()
    plt.clf()

def plot_informations_over_clusters(data, opt, save_dir):
    plt.figure(figsize=(20, 9))
    x_range = np.arange(opt.k_0, opt.k_fin)
    file_name = 'NMI_interK_'
    plt.suptitle(
        f'Plot of NMI between Clusters configuration with three different means: \n'
        f' Arithmetic, Harmonic, Median. ', fontsize=25, y=0.99
        ,
        fontweight="bold"
    )
    NMI_matrix = deepcopy(data)
    y = [NMI_matrix[NMI_matrix[col] != 0][[col]].mean() for col, x in zip(NMI_matrix.columns, x_range)]
    harmonic_mean = [len(NMI_matrix[NMI_matrix[col] != 0][[col]]) / np.sum(1 / NMI_matrix[NMI_matrix[col] != 0][[col]]) for col, x in zip(NMI_matrix.columns, x_range)]
    median = [np.median(NMI_matrix[NMI_matrix[col] != 0][[col]]) for col, x in zip(NMI_matrix.columns, x_range)]
    std = [np.std(NMI_matrix[NMI_matrix[col] != 0][[col]]) for col, x in zip(NMI_matrix.columns, x_range)]
    plt.plot(x_range, y, linewidth=6, color='r', label='mean')
    plt.plot(x_range, harmonic_mean, linewidth=6, color='k', label='harmonic mean')
    plt.plot(x_range, median, linewidth=6, color='g', label='median')
    plt.legend(fontsize=25)
    plt.xticks(x_range, [f'K_{x_} -> K_{x_ + 1}' for x_ in x_range], fontsize=15)
    plt.yticks(color='k', fontsize=15)
    plt.grid()
    plt.savefig(os.path.join(save_dir, '{}___.png'.format(file_name)))
    plt.close()
    plt.clf()




def plot_metrics_unsupervised_EXPs_mean_var(data, save_dir, Ks):
    file_name = "Metrics_Unsupervised_over_Exps"
    K_ = np.array(Ks)
    keys = list(data.keys())
    tabs = ["r", "", "k", "", "g", ""]
    for i in range(0, len(keys), 2):
        key = keys[i]
        data_plot = data[keys[i]]
        var_ = data[keys[i + 1]]
        plt.clf()
        plt.figure(figsize=(20, 9))
        plt.errorbar(
            x=K_, y=data_plot, yerr=var_, color=tabs[i], linestyle='None', linewidth=3, fmt='-o', label='St-Dev' + keys[i].split('_')[0]
        )
        plt.plot(
            K_, data_plot, tabs[i], label='Mean ' + keys[i].split('_')[0], linewidth=6, )
        plt.xticks(K_, fontsize=22)
        plt.yticks(color='k', fontsize=22)
        plt.xlabel('Number of Clusters', fontsize=25)
        plt.ylabel(key, fontsize=25, color="k", fontweight="bold")
        plt.grid()
        plt.legend(fontsize=25)
        plt.suptitle(
            f'Plot of {keys[i]} and Standard Deviation over Experiments', fontsize=25, y=0.91,
            fontweight="bold"
        )
        plt.savefig(os.path.join(save_dir, '{}___{}.png'.format(file_name, keys[i].split('_')[0])))
        plt.close()
        plt.clf()

        import numpy as np
        np.harmoni
def plot_metrics_unsupervised_K(file, save_dir):
    file_name = f"Metrics_Unsupervised_over_k"
    data = pd.read_csv(file)
    keys = data.keys()
    K_ = data[keys[0]]
    tabs = ["r", "k", "g"]
    KEYS_ = {keys[1]: "Silhouette", keys[2]: "Calinski Harabasz", keys[3]: "Davies Bouldin"}
    for i in range(len(data.keys()) - 1):
        data_plot = data[keys[i + 1]]
        plt.figure(figsize=(15, 10))
        plt.plot(
            K_, data_plot, tabs[i], label=KEYS_[keys[i + 1]] + f'\n Valore massimo := '
                                                               f'{(np.round(np.max(data[keys[i + 1]]), 4))}, '
                                                               f'\n Valore Minimo :=  '
                                                               f'{(np.round(np.min(data[keys[i + 1]]), 4))}   ',

            linewidth=4, )
        plt.xticks(K_, fontsize=22)
        plt.yticks(color='k', fontsize=22)
        plt.xlabel('k di inizializzazione', fontsize=25)
        plt.ylabel(KEYS_[keys[i + 1]], fontsize=25, color="k", fontweight="bold")
        plt.grid()
        plt.legend(fontsize=25)
        plt.suptitle(
            f'Andamento {KEYS_[keys[i + 1]]} al variare di k di inizializzazione', fontsize=25, y=0.91,
            fontweight="bold"
        )
        plt.savefig(os.path.join(save_dir, '{}___{}.png'.format(file_name, keys[i + 1])))
        plt.clf()
        plt.close()


def show_labeled_data(X_l_sel, select_label, ids_lab_sel, save_dir, file_name, number_to_plot=25):
    import math
    len_ = len(X_l_sel)
    plt.clf()
    number_to_plot = number_to_plot if not len_ <= number_to_plot else len_
    if not len_ == 0:
        step_ = len_ // number_to_plot
        fig = plt.figure(figsize=(10, 10))
        rows = math.ceil(math.sqrt(number_to_plot))
        columns = math.ceil(math.sqrt(number_to_plot))
        to_plot = np.arange(0, len_, step_)
        fig.suptitle(f'Plotted Images for Label Selected = {select_label + 1}')
        for i in range(0, number_to_plot):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(X_l_sel[to_plot[i]], cmap='gray')
            plt.axis('off')
            plt.title(ids_lab_sel[to_plot[i]])
        fig.savefig(os.path.join(save_dir, '{}_label_sel_{}___.png'.format(file_name, select_label + 1)))
        plt.close(fig)
    else:
        print("Empty_CLuster")
