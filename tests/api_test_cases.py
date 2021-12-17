"""
Auxiliary test cases.
The idea is to move a lot of the in api_test.py `user_input_case` function
to here.

For now not used.
"""
# <=50k
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

# >50k
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
