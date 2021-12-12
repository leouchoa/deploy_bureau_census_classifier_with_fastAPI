"""
Functions for evaluating model performance.
"""
# import model
# import utils as ut
# from data import process_data
import os
import pickle as pkl
from typing import Dict

import pandas as pd
from ml import model
from ml import utils as ut
from ml.data import process_data
from sklearn.metrics import fbeta_score, precision_score, recall_score


def compute_model_metrics(true_y, predicted_y):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(true_y, predicted_y, beta=1, zero_division=1)
    precision = precision_score(true_y, predicted_y, zero_division=1)
    recall = recall_score(true_y, predicted_y, zero_division=1)
    return precision, recall, fbeta


def load_model_and_processors(model_path: str):
    """
    From given path, loads:
        - Model
        - LabelBinarizer
        - OneHotEncoder
    **It is assumed that they are all in the same path**
    -------
    Inputs:
    -------
        - model_path: path to load all *pickled* objects
    -------
    Outputs:
    -------
        - pickled ojects for
            - Model
            - LabelBinarizer
            - OneHotEncoder
    """
    clf_path = os.path.join(model_path, "classifier.pkl")
    encoder_path = os.path.join(model_path, "encoder.pkl")
    lb_path = os.path.join(model_path, "label_binarizer.pkl")

    assert os.path.exists(model_path), f"{model_path} does not exists"

    out = []

    for obj_path in [clf_path, encoder_path, lb_path]:
        try:
            with open(obj_path, "rb") as f:
                out.append(pkl.load(f))
        except FileNotFoundError:
            print(f"{obj_path} doest not exists")

    return out


def model_metrics_by_slice(
    target_df, clf, encoder, lb, categorical_features, save_to: str = None
) -> pd.DataFrame:
    """
    -------
    Inputs:
    -------
        - target_df: a
        - clf: classifier model
        - encoder: one-hot-encoder (input to data.process_data)
        - lb: label binarizer (input to data.process_data)
        - categorical_features: categorical features used
        - save_to: path to save resulting data.frame, if desired
    -------
    Outputs:
    -------
        - pd.Dataframe with model metrics by every categorical_features category.
        - save the results into `save_to/overall_metrics.txt`
    """
    res_columns = ["cat_feat", "category", "precision", "recall", "fbeta"]

    res = pd.DataFrame(columns=res_columns)

    for cat_feat in categorical_features:
        for category in target_df[cat_feat].unique():

            filtered_df = target_df[target_df[cat_feat] == category]

            X_filtered, y_filtered, *_ = process_data(
                X=filtered_df,
                categorical_features=categorical_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )

            preds = model.inference(clf, X_filtered)

            aux = compute_model_metrics(true_y=y_filtered, predicted_y=preds)

            res = res.append(
                pd.DataFrame(
                    {
                        "cat_feat": {"": f"{cat_feat}"},
                        "category": {"": f"{category}"},
                        "precision": {"": f"{aux[0]}"},
                        "recall": {"": f"{aux[1]}"},
                        "fbeta": {"": f"{aux[2]}"},
                    }
                )
            )

    if save_to is not None:
        res.to_csv(
            os.path.join(save_to, "model_metrics_by_slice.csv"), index=False
        )

    return res


def overall_model_metrics(
    target_df, clf, encoder, lb, categorical_features, save_to: str = None
) -> Dict:
    """
    Compute the overall (non-sliced) model metrics:
        - fbeta
        - recall
        - precision
    Can also save the results into `save_to/overall_metrics.txt`
    -------
    Inputs:
    -------
        - target_df: a
        - clf: classifier model
        - encoder: one-hot-encoder (input to data.process_data)
        - lb: label binarizer (input to data.process_data)
        - categorical_features: categorical features used
        - save_to: path to save resulting data.frame, if desired
    -------
    Outputs:
    -------
        - Dictionary containing:
            - fbeta
            - recall
            - precision
        - Can be saved into `save_to/overall_metrics.txt`
    """
    X_eval, y_eval, *_ = process_data(
        X=target_df,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = model.inference(clf, X_eval)

    overall_metrics = compute_model_metrics(true_y=y_eval, predicted_y=preds)

    out = {
        "precision": f"{overall_metrics[0]}",
        "recall": f"{overall_metrics[1]}",
        "fbeta": f"{overall_metrics[2]}",
    }

    if save_to is not None:
        save_to = os.path.join(save_to, "overall_metrics.txt")

        with open(save_to, "w") as f:
            f.write(str(out))

    return out


if __name__ == "__main__":
    data_path = "../../data/train_test_split/test_df.csv"

    test_data = pd.read_csv(data_path)

    clf, encoder, lb = load_model_and_processors("../../model")

    model_metrics_by_slice(
        target_df=test_data,
        clf=clf,
        encoder=encoder,
        lb=lb,
        categorical_features=ut.get_cat_features(),
        save_to="../../model",
    )

    overall_model_metrics(
        target_df=test_data,
        clf=clf,
        encoder=encoder,
        lb=lb,
        categorical_features=ut.get_cat_features(),
        save_to="../../model",
    )
