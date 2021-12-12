# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Created by Leonardo Uchôa Pedreira, the model is a Random Forest Classifier, trained with the scikit learn python library. For now it was trained with default parameters provided by scikit learn.

## Intended Use

The model was created with the intention to be used as a predictor of salary , given some personal socioeconomic characteristics.

## Training Data

### Basic Data Information

The raw data used to train the model, along with more information about the features, can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).

This data was downloaded and treated to remove white spaces, being called `cleaned_data.csv`

### Data Splitting
To train the model, a train/test split strategy was employed with 80% being the training size and 20% for testing/evaluation. 

### Pre-processing

To encode the training data to the appropriate modeling format, two transformations were applied:

- [Binarization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) of the salary (target) variable.
- [One-hot-encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) of categorical features.

Those two transformations were trained on training data and also saved to further utilization.

## Evaluation Data

The evaluation (test) data is composed of the remaining 20% of train/test split. The evaluation data also had to pass through the pre-processing step to correctly encode it as model-appropriate dataset.

## Metrics

The metrics used to evaluate the model were [precision, recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) and [fbeta](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html). The overall performance of the model at the test dataset are:

- precision: 0.73
- recall: 0.63
- fbeta: 0.68

For more detailed information of how the model performs on slices of data can be found in `model/model_metrics_by_slice.csv`

## Ethical Considerations

Personal characteristics like gender, race, and native country are present at the dataset and those can potentially act as source of social discrimination and therefore social bias. So the model should be taken into account when used to make critical conclusions or costumer services. 

It must be also noted that many countries have low representativity in the dataset because of low amount of observations. So it is expected that model performs better on US, mostly because of the amount of data present, altought such a hypothesis was not tested.

## Caveats and Recommendations

Some characteristics, like the target feature salary, are heavily unbalanced and ethical and bias may arise. Because of that it's advised do further balance the data with stratefied sampling. Another sugestion is to do hyperparameter optimization to improve model performance and also conduct an interpretability analysis with tools like SHAP.

