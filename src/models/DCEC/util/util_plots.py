
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
    fig1.suptitle('Plot APN, NMI, e metrica combinata $\mathbf{\sqrt{NMI_{01} * APN_{01}}}$', fontsize=28, y=0.94,
        fontweight='bold')
    plt.yticks(fontsize=22)
    fig1.savefig(os.path.join(save_dir, '{}___.png'.format(file_name)))
    plt.close(fig1)

def plt_Var_Gini_K(file, save_dir):
    data = pd.read_csv(file)
    keys = data.keys()
    K_ = np.array(data["K"])
    matrix_gini = np.array([tuple(map(float, i[1:-1].split(', '))) for i in data["Gini"]])
    matrix_Var = np.array([tuple(map(float, i[1:-1].split(', '))) for i in data["Var"]])
    matrix_Var_w = np.array([tuple(map(float, i[1:-1].split(', '))) for i in data["Var_w"]])

    N_patients = matrix_Var.shape[1]
    N_K = K_.shape[0]
    tabs = ["r", "k"]
    # _________________________________________________________________________________________________________________
    # Plot Varianza

    # _________________________________________________________________________________________________________________
    # Plot Varianza Pesata
    N_Var_w_matrix, Delta_Var_w_matrix, D_sum_Var_w_matrix = util_clustering.calc_Delta_metrics(matrix_Var_w,
        N01=False)
    file_name = f"Mean_Var_w_over_k",
    plt.clf()
    plt.close()
    plt.figure(1, figsize=(25, 12))

    data_plot = N_Var_w_matrix.copy()
    if data_plot.shape[1] > 20:
        print("here")
    else:
        for i in range(data_plot.shape[1]):
            plt.plot(K_, data_plot[:, i], linewidth=3, linestyle="dashdot", label=" digit " + str(i))
            plt.legend(fontsize=20)

    plt.plot(K_, np.array([np.mean(data_plot[i, :]) for i in range(N_K)]), tabs[0],
        label="$mean_{t \in T} VAR_wt$ al variare di k", linewidth=5, )
    plt.xticks(K_, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    plt.xlabel('k di inizializzazione', fontsize=25)
    plt.ylabel("$\mathbf{mean_{t \in T} VAR_wt}$", color="k", position=[-1, 0], fontsize=25, )
    plt.suptitle('Plot dell\'andamento $\mathbf{mean_{t \in T} VAR_wt}$ al variare di k ', fontsize=28, y=0.94,
        fontweight='bold')
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
        fontsize=28, y=0.94, fontweight='bold')
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.close()
    # _________________________________________________________________________________________________________________
    # Plot GINI

    Gini, Delta_Gini, Delta_Gini_Sum = utils_clustering.calc_Delta_metrics(matrix_gini, N01=False)

    file_name = f"Gini_mean_t_over_k",
    data_plot = Gini.copy()
    plt.figure(3, figsize=(25, 12))
    plt.clf()
    if data_plot.shape[1] > 20:
        print("here")
    else:
        for i in range(data_plot.shape[1]):
            plt.plot(K_, data_plot[:, i], linewidth=3, linestyle="dashdot", label=" digit " + str(i))
            plt.legend(fontsize=20)
    plt.plot(K_, np.array([np.mean(data_plot[i, :]) for i in range(N_K)]), tabs[0],
        label="$\mathbf{mean_{t \in T} GINI_t}$ al variare di k", linewidth=5, color="b")
    plt.xticks(K_, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    plt.xlabel('k di inizializzazione', fontsize=25)
    plt.ylabel("$\mathbf{mean_{t \in T} GINI_t}$", color="k", position=[-1, 0], fontsize=25)
    plt.suptitle('Plot dell\'andamento $\mathbf{mean_{t \in T} GINI_t}$ al variare di k ', fontweight='bold',
        fontsize=28, y=0.94)
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
        fontsize=28, y=0.94, fontweight='bold')
    plt.savefig(os.path.join(save_dir, '{}__.png'.format(file_name)))
    plt.close()
    plt.clf()