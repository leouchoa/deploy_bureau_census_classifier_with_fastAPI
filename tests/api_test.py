"""
Testing the FastAPI for serving a salary predictor.
Tests covered:
- test greeting message
- test API response code
- test API salary category output
"""
import pytest
from fastapi.testclient import TestClient

import modeling_pipeline.placeholder as inference_utils
from app import app
from modeling_pipeline.ml.evaluate_model import load_model_and_processors
from modeling_pipeline.ml.utils import get_cat_features

client = TestClient(app)


@pytest.fixture(scope="session")
def user_input_case():
    user_input_1 = {
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    user_input_2 = {
        "age": 42,
        "workclass": "Private",
        "education": "Masters",
        "marital_status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 55,
        "native_country": "United-States",
    }
    return (user_input_1, user_input_2)


# This is needed because otherwise the following error will appear:
# --------------------------------------------------------------------------------------
# AttributeError: module 'modeling_pipeline.placeholder' has no attribute 'cat_features'
# --------------------------------------------------------------------------------------
# And it's because the app.on_event("startup") handler is not activated,
# which means that the `load_model` function won't work.
# So it's necessary to load the pre-processors like that so that they're
# available within `inference_utils`
clf, encoder, lb = load_model_and_processors("model")
cat_features = get_cat_features()
inference_utils.clf = clf
inference_utils.encoder = encoder
inference_utils.lb = lb
inference_utils.cat_features = cat_features


def test_read_main():
    """
    Test greetings message
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome!"}


def test_prediction_status_code(user_input_case):
    """
    Test correct predictions response code
    """
    user_input_1, _ = user_input_case
    response = client.post(
        "http://localhost:8000/predictions",
        json=user_input_1,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200


def test_prediction_response_category(user_input_case):
    """
    Test correct predictions output
    """
    user_input_1, user_input_2 = user_input_case

    response_1 = client.post(
        "http://localhost:8000/predictions",
        json=user_input_1,
        headers={"Content-Type": "application/json"},
    )

    response_2 = client.post(
        "http://localhost:8000/predictions",
        json=user_input_2,
        headers={"Content-Type": "application/json"},
    )

    with open("/tmp/asd.txt", "w") as f:
        f.write(f"{response_1.json()} \n {response_2.json()}")

    assert response_1.json() == {"prediction": "<=50K"}
    assert response_2.json() == {"prediction": ">50K"}
