"""
Code for creating an API with FastAPI. The requests an user can make is to
ask for predictions in a given dataset.

For now the code only supports online predictions.
"""
import pandas as pd
from fastapi import FastAPI

import modeling_pipeline.placeholder as inference_utils
from modeling_pipeline.ml.data import process_data
from modeling_pipeline.ml.evaluate_model import load_model_and_processors
from modeling_pipeline.ml.model import get_original_y_class, inference
from modeling_pipeline.ml.utils import get_cat_features
from web_api.census_class import Census

app = FastAPI(
    title="Salary Classifier API",
    description="An API for classifiying salaries",
    version="0.0.1",
)


# greeting message =)
@app.get("/")
def greeting():
    """
    Home welcome message
    """
    return {"message": "Welcome!"}


# makes sure that FastAPI imports the model only when
# the app gets started and not in every request
# source:
# https://medium.com/analytics-vidhya/serve-a-machine-learning-model-using-sklearn-fastapi-and-docker-85aabf96729b
@app.on_event("startup")
async def load_model():
    """
    This is an event handler that loads the model only when the API is
    started, instead of loading it after every request.
    """
    clf, encoder, lb = load_model_and_processors("model")
    cat_features = get_cat_features()
    inference_utils.clf = clf
    inference_utils.encoder = encoder
    inference_utils.lb = lb
    inference_utils.cat_features = cat_features


@app.post("/predictions", tags=["predictions"])
async def make_prediction(user_input: Census):
    """
    Prediction request.
    Users can get predictions from the model by using this POST request.

    Input:

    [docs](http://127.0.0.1:8000/docs#/predictions/make_prediction_predictions_post)

    Output:

        Classification of salary into either less than 50k or more than 50k:
        - <=50k
        - >50k
    """
    user_input = pd.DataFrame(
        {k: v for k, v in user_input.dict().items()}, index=[0]
    )

    X_eval, *_ = process_data(
        X=user_input,
        categorical_features=inference_utils.cat_features,
        label=None,
        training=False,
        encoder=inference_utils.encoder,
        lb=inference_utils.lb,
    )

    # -------------
    # !!!!!IMPORTANT!!!!!
    # -------------
    # You must use `.tolist()[0]` after making predictions, otherwise you'll receive a
    # myriad of errors, like:
    # - TypeError: cannot convert dictionary update sequence element #0 to a sequence
    # - TypeError: vars() argument must have __dict__ attribute
    #
    # And you'll receive the error even tough predictions goes as expected, and
    # you can test it with:
    #
    # with open("/tmp/pred_test.txt", "w") as f:
    #     f.write(f"first = {str(inference_utils.clf.predict(X_eval))}")
    #     f.write("\n")
    #     f.write(f"second = {str(inference(model=inference_utils.clf, X=X_eval))}")
    #
    #
    # Both of those versions work well:
    #
    # preds = inference(model=inference_utils.clf, X=X_eval).tolist()[0]
    # preds = inference_utils.clf.predict(X_eval).tolist()[0]
    # preds = inference(model=inference_utils.clf, X=X_eval).tolist()[0]
    # notice that we're slicing the first element because that's not
    # batch inference
    #
    preds = inference(model=inference_utils.clf, X=X_eval)
    orig_class_pred = get_original_y_class(
        pred=preds, label_binarizer=inference_utils.lb
    )[0]
    return {"prediction": orig_class_pred}


# if __name__ == '__main__':

#     uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)
