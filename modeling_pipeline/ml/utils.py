"""
Some Util functions
"""
from typing import List

import pandas as pd


def load_data(path_to_data: str) -> pd.DataFrame:
    """
    loads csv file from give path
    Inputs:
        - path_to_data
    Outputs:
        - pd.DataFrame
    """
    df = pd.read_csv(path_to_data)
    return df


def get_cat_features() -> List:
    """
    Return categorical features
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_features
