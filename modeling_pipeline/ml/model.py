"""
Utility functions for:
- Training models
- Making inference
"""
import os
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier as rf


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = rf()
    clf.fit(X_train, y_train)

    return clf


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : Random Forest Classifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(path_to_save: str, clf) -> None:
    """
    Saves trained classifier model
    -------
    Inputs
    -------
    model : Random Forest Classifier
        Trained machine learning model.
    path_to_save : str
        Path to save model
    Returns
    -------
    Saved model into `path_to_save/classifier.pkl`
    """
    path_to_save = os.path.join(path_to_save, "classifier.pkl")
    pkl.dump(clf, open(path_to_save, "wb"))
