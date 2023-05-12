import itertools
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt



def plot_training_acc_and_Combined_ROC_CURVE(X, Y, classifier, kf, Normalizing= False, Experiment_name= str):

    tp_rates = []
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









def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color
    # Confusion matrix
    # One can just simply type confusion_matrix(y_test, y_pred)
    # to get the confusion matrix. However, let’s take a more advanced approach.
    # Here, I create a function to plot confusion matrix, which prints and plots
    # the confusion matrix.
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('True label', size = 18)
        plt.xlabel('Predicted label', size = 18)

def evaluate_model_(y_pred, y_test, probs_test):
    baseline = {}
    baseline['recall']=recall_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs_test)
    for metric in ['recall', 'precision', 'roc']:
          print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} ')
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs_test)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()


