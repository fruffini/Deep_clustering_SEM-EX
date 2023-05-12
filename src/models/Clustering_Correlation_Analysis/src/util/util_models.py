from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from pytorch_tabnet.tab_model import TabNetClassifier


def _import_model(model_type=str):
    try:
        assert model_type.__len__() > 1
        kwargs = {'random_state': 42}
        if model_type == 'RF':
            return RandomForestClassifier(random_state=42)
        elif model_type == 'AdaBoost':
            return AdaBoostClassifier(random_state=42)
        elif model_type == 'SVM':
            return SVC(random_state=42, probability=True)
        elif model_type == 'DT':
            return DecisionTreeClassifier(random_state=42)
        elif model_type == "MLP":
            return MLPClassifier(random_state=42)
        elif model_type == "TabNet":
            return TabNetClassifier()
        else:
            raise Exception
    except:
        print(f"The model has not been choosen")


def _train_classifier(classifier, X_train, y_train, X_test):
    classifier.fit(X_train.values, y_train.values)
    probs_ = classifier.predict_proba(X_test.values)
    y_pred = classifier.predict(X_test.values)
    return probs_, y_pred
