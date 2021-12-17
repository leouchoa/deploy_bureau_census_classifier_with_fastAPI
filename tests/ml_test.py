"""
Test suite for functions in `ml/`
"""
import os
import pickle as pkl

import pandas as pd
import pytest

import modeling_pipeline.ml.data as data
import modeling_pipeline.ml.evaluate_model as eval
import modeling_pipeline.ml.model as model_utils
import modeling_pipeline.ml.utils as ut

TEST_SIZE = 0.2
PATH_TO_DATA = "data/census_cleaned.csv"
PATH_TO_DATA_TRAIN_TEST_DF = "data/"
PATH_TO_SAVE_MODEL = "model/"
LABEL = "salary"


@pytest.fixture(scope="session")
def df():
    """
    Load dataset and attach it to session with fixture
    """
    df = pd.read_csv("data/census_cleaned.csv")
    return df


@pytest.fixture(scope="session")
def train_test_dfs():
    """
    Load train/test datasets and attach it to session with fixture
    """
    train_df = pd.read_csv("data/train_test_split/train_df.csv")
    test_df = pd.read_csv("data/train_test_split/test_df.csv")

    return (train_df, test_df)


@pytest.fixture(scope="session")
def transformers_and_model():
    """
    Loads encoder, label_binarizer, classifier model and attach
    it to session with fixture
    """
    encoder = pkl.load(
        open(os.path.join(PATH_TO_SAVE_MODEL, "encoder.pkl"), "rb")
    )

    lb = pkl.load(
        open(os.path.join(PATH_TO_SAVE_MODEL, "label_binarizer.pkl"), "rb")
    )

    clf = pkl.load(
        open(os.path.join(PATH_TO_SAVE_MODEL, "classifier.pkl"), "rb")
    )

    return (encoder, lb, clf)


def test_encoder_and_lb(train_test_dfs, transformers_and_model):
    """
    Test if encoder and label binarizer doesnt mess up with number of rows
    """
    _, test_df = train_test_dfs

    encoder, lb, _ = transformers_and_model

    X_test, y_test, _, _ = data.process_data(
        test_df,
        categorical_features=ut.get_cat_features(),
        label="salary",
        encoder=encoder,
        lb=lb,
        training=False,
    )

    assert len(X_test) == len(y_test), "Number of rows of X and y differ"


def test_processor_results(df, train_test_dfs, transformers_and_model):
    """
    Test if new encoder and label binarizer doesnt mess up with
    shape of dataframe
    """
    train_df, test_df = train_test_dfs

    encoder, lb, _ = transformers_and_model

    *_, encoder_new, lb_new = data.process_data(
        X=train_df,
        categorical_features=ut.get_cat_features(),
        label="salary",
        training=True,
    )

    X_eval_old_transformers, y_eval_old_transformers, *_ = data.process_data(
        X=test_df,
        categorical_features=ut.get_cat_features(),
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    X_eval_new_transformers, y_eval_new_transformers, *_ = data.process_data(
        X=test_df,
        categorical_features=ut.get_cat_features(),
        label="salary",
        training=False,
        encoder=encoder_new,
        lb=lb_new,
    )

    assert X_eval_new_transformers.shape == X_eval_old_transformers.shape, (
        "New One-Hot-Encoder resulting data-frame dimensions differ"
        "from original"
    )

    assert y_eval_new_transformers.shape == y_eval_old_transformers.shape, (
        "New Label Binarizer resulting data-frame dimensions differ"
        "from original"
    )


def test_model_metrics_by_slice_correct_number_of_groups(
    train_test_dfs, transformers_and_model
):
    """
    Test if model_metrics_by_slice computes model metrics for
    every category in every categorical feature.
    """
    _, test_df = train_test_dfs

    encoder, lb, clf = transformers_and_model

    cat_features = ut.get_cat_features()

    total_categories = []

    for cat_feat in cat_features:
        total_categories.append(len(test_df[cat_feat].unique()))

    total_categories = sum(total_categories)

    res = eval.model_metrics_by_slice(
        target_df=test_df,
        clf=clf,
        encoder=encoder,
        lb=lb,
        categorical_features=ut.get_cat_features(),
        save_to=None,
    )

    assert len(res) == total_categories, (
        "There's at least one category left behind in"
        "`model_metrics_by_slice`"
    )


def test_inference_nrows(train_test_dfs, transformers_and_model):
    """
    Test if inference function outputs correct number of observations
    """
    _, test_df = train_test_dfs

    encoder, lb, clf = transformers_and_model

    X_test, y_test, *_ = data.process_data(
        X=test_df,
        categorical_features=ut.get_cat_features(),
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = model_utils.inference(model=clf, X=X_test)

    assert len(preds) == len(
        y_test
    ), "Nrows of predictions differ from y_test: " "{} != {}".format(
        preds, y_test
    )
