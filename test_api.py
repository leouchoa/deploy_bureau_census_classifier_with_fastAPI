"""
Test the heroku deployed salary classifier
"""

import json

import requests

test_case = {
    "example": {
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
}

r1 = requests.get("https://salary-classifier.herokuapp.com")

r2 = requests.post(
    "https://salary-classifier.herokuapp.com", json=json.dumps(test_case)
)

assert r1.status_code == 200
assert r2.status_code == 200

print(f"Response body: {r2.json()}")
