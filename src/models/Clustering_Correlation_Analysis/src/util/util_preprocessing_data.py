import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import accuracy_score, roc_curve, auc
from src.models.Clustering_Correlation_Analysis.src.util.util_models import train_classifier


def plot_training_acc_and_Combined_ROC_CURVE(X, Y, classifier, kf, Normalizing= False, Experiment_name= str):

    tprs = []
    aucs = []
    accuracies = dict()
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 10))
    # We spilt each FOLD by index
    for i, (train, test) in enumerate(kf.split(X, Y)):
        # normalization
        X_train, max_, min_  = normalize(X.iloc[train]) if Normalizing else X.iloc[train], 0, 0
        X_test, _, _ = normalize(X.iloc[test], max_ = max_, min_ = min_) if Normalizing else X.iloc[test], 0, 0

        # Labels
        y_train = Y.iloc[train]
        y_test = Y.iloc[test]

        # Fill NaN
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)


        # Train the model
        probs_, y_pred = train_classifier(classifier, X_train, y_train, X_test)

        # Accuracy Scores
        accuracies[i] = accuracy_score(y_test, y_pred)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probs_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(
            fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc)
            )

        i += 1
    plt.plot(
        [0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8
        )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(
        mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=4, alpha=.8,
        )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        label=r'$\pm$ 1 std. dev.'
        )

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f'Cross-Validation ROC : {Experiment_name}', fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.show()
    return (accuracies, classifier)


def normalize(df, min_=None, max_=None):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max() if max_ is None else max_
        min_value = df[feature_name].min() if min_ is None else min_

        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result, max_value, min_value