# deploy_bureau_census_classifier_with_fastAPI
deploy_bureau_census_classifier_with_fastAPI

## Context

Develop a classification model on [publicly available Census Bureau data](https://archive.ics.uci.edu/ml/datasets/census+income) and deploy an API with [heroku](https://www.heroku.com/) that returns model predictions.

## How to use

You can go to the [api serving website](https://salary-classifier.herokuapp.com/docs) and try it out, there's an example there.

## How it works

The app pulls the [tracked model](https://dvc.org/) from an [Amazon S3 bucket](https://aws.amazon.com/pt/s3/) and loads it to the [API](https://fastapi.tiangolo.com/) to make predictions. 

So if the app breaks, probably it's because the S3 storage provided by the [course](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) was interrupted.

## About the model

There's a [model card](https://modelcards.withgoogle.com/about) for you to understand better about the model performance and use cases, along with ethical considerations.

## CI/CD

All the code and model passes through an CI/CD lifecycle in order to ensure realiability of the deployed service. Tests are stored in the [tests folder](https://github.com/leouchoa/deploy_bureau_census_classifier_with_fastAPI/tree/main/tests) and are automatically run whenever a push to the main branch occours. So the main branch is the deployed branch and all development must occour outside of that branch.

Besides that there's a continuous deployment configured in heroku. Everytime a there's a commit in the main branch, heroku waits for the CI tests to pass to only them re-deploy the app, so the app is only updated whenever the code passes all the tests.


### Pre-Commit

There's pre-commit file in this repo. You can install it by:

1. installing pre-commit in python with `pip install pre-commit`
2. installing the configuration with `pre-commit install`

For more info, check [this repo](https://github.com/leouchoa/actions_test).
