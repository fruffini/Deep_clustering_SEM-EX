






def train_classifier(classifier, X_train, y_train, X_test):
    probs_ = classifier.fit(X_train.values, y_train.values).predict_proba(X_test.values)
    y_pred = classifier.predict(X_test.values)
    return probs_, y_pred