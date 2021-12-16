"""
Utility functions for:
- Training models
- Making inference
"""
import os
import pickle as pkl
from typing import List

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


def get_original_y_class(pred: int, label_binarizer) -> List:
    """
    Revert from 0/1 to original y class <=50k/>50k.
    Meant to be specially used within the API for convenience reasons
    """
    return label_binarizer.inverse_transform(pred).tolist()


if __name__ == "__main__":
    import pandas as pd

    from modeling_pipeline.ml.data import process_data
    from modeling_pipeline.ml.evaluate_model import load_model_and_processors
    from modeling_pipeline.ml.utils import get_cat_features

    data_path = "data/train_test_split/test_df.csv"

    test_data = pd.read_csv(data_path)

    clf, encoder, lb = load_model_and_processors("model")

    X_eval, *_ = process_data(
        X=test_data,
        categorical_features=get_cat_features(),
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(clf, X_eval)

    print(get_original_y_class(pred=preds, label_binarizer=lb))
