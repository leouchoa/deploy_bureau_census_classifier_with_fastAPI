"""
Main function for model training. Steps executed are:
- Load data
- Make (and possibly save) train/test split
- Preprocess data and save pre-processors
- Train and save model
- Compute and save overall/sliced model metrics
"""
from ml import evaluate_model as eval
from ml import model
from ml import utils as ut
from ml.data import process_data, save_data_processors, split_and_save
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
PATH_TO_DATA = "../data/census_cleaned.csv"
PATH_TO_DATA_TRAIN_TEST_DF = "../data/"
PATH_TO_SAVE_MODEL = "../model/"
LABEL = "salary"


def train_model(
    cleaned_data,
    test_size,
    path_to_save_model,
    path_to_save_train_test_df,
    save_train_test_df=False,
) -> None:
    """
    Main function to orchestrate the steps:
        - Load data
        - Make (and possibly save) train/test split
        - Preprocess data and save pre-processors
        - Train and save model
        - Compute and save overall/sliced model metrics
    """
    if save_train_test_df:
        train_df, test_df = split_and_save(
            cleaned_data=cleaned_data,
            test_size=test_size,
            path_to_save_train_test_df=path_to_save_train_test_df,
        )
    else:
        train_df, test_df = train_test_split(cleaned_data, test_size=test_size)

    X_train, y_train, encoder, lb = process_data(
        X=train_df,
        categorical_features=ut.get_cat_features(),
        label="salary",
        training=True,
    )

    X_test, y_test, *_ = process_data(
        X=test_df,
        categorical_features=ut.get_cat_features(),
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    save_data_processors(
        path_to_save=path_to_save_model, encoder=encoder, lb=lb
    )

    clf = model.train_model(X_train=X_train, y_train=y_train)

    model.save_model(path_to_save=path_to_save_model, clf=clf)

    eval.model_metrics_by_slice(
        target_df=X_test,
        clf=clf,
        encoder=encoder,
        lb=lb,
        categorical_features=ut.get_cat_features(),
        save_to=path_to_save_model,
    )

    eval.overall_model_metrics(
        target_df=X_test,
        clf=clf,
        encoder=encoder,
        lb=lb,
        categorical_features=ut.get_cat_features(),
        save_to=path_to_save_model,
    )


if __name__ == "__main__":
    df = ut.load_data(PATH_TO_DATA)

    train_model(
        cleaned_data=df,
        test_size=TEST_SIZE,
        path_to_save_train_test_df=PATH_TO_DATA_TRAIN_TEST_DF,
        path_to_save_model=PATH_TO_SAVE_MODEL,
        save_train_test_df=True,
    )
