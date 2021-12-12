"""
Data wrangling and pre-processing related functions for:

- Pre-processing
- Train/test split and save
- Pre-processors save

"""
import os
import pickle as pkl
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=[],
    label=None,
    training=True,
    encoder=None,
    lb=None,
):
    """Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def split_and_save(
    cleaned_data: pd.DataFrame,
    test_size: float,
    path_to_save_train_test_df: str,
) -> Tuple:
    """
    Make train/test split and also save the resulting dataframes to path
    -------
    Inputs:
    -------
        - cleaned_data: white spacing trimmed version of census.csv
        - test_size: split percentage allocated to test dataframe
        - path_to_save_train_test_df: path to save both train.csv and test.csv
    -------
    Ouputs:
    -------
        - both train.csv and test.csv saved into
            os.path.join(path_to_save_train_test_df,"train_test")
    """
    train_df, test_df = train_test_split(cleaned_data, test_size=test_size)

    path_to_save_train_test_df = os.path.join(
        path_to_save_train_test_df, "train_test_split"
    )

    if not os.path.exists(path_to_save_train_test_df):
        os.makedirs(path_to_save_train_test_df)

    train_df.to_csv(
        os.path.join(path_to_save_train_test_df, "train_df.csv"), index=False
    )

    test_df.to_csv(
        os.path.join(path_to_save_train_test_df, "test_df.csv"), index=False
    )

    return (train_df, test_df)


def save_data_processors(path_to_save: str, encoder, lb) -> None:
    """
    Save one_hot_encoder (encoder) and label_binarizer (label) to folder
    -------
    Inputs:
        - path_to_save: path to where preprocessors will be saved
        - encoder: OneHotEncoder
        - lb: LabelBinarizer
    -------
    Outputs:
    -------
    Saved pickled objects into:
        - path_to_save/encoder.pkl
        - path_to_save/label_binarizer.pkl
    """
    path_to_encoder = os.path.join(path_to_save, "encoder.pkl")

    path_to_lb = os.path.join(path_to_save, "label_binarizer.pkl")

    pkl.dump(encoder, open(path_to_encoder, "wb"))

    pkl.dump(lb, open(path_to_lb, "wb"))
