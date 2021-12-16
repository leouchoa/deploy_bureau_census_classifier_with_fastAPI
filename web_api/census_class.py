"""
POST method creation and documentation of it
"""
from typing import Literal

from pydantic import BaseModel


# thanks to:
#
# https://github.com/ximenesfel
#
# for the hard work
# ------
# DO NOT USE `int`, USE `float`. It will give the error described here
# https://stackoverflow.com/questions/65450684/converting-json-into-a-dataframe-within-fastapi-app
class Census(BaseModel):
    """
    Input documentation in FastAPI POST method
    """

    age: float
    workclass: Literal[
        "State-gov",
        "Self-emp-not-inc",
        "Private",
        "Federal-gov",
        "Local-gov",
        "Self-emp-inc",
        "Without-pay",
    ]
    education: Literal[
        "Bachelors",
        "HS-grad",
        "11th",
        "Masters",
        "9th",
        "Some-college",
        "Assoc-acdm",
        "7th-8th",
        "Doctorate",
        "Assoc-voc",
        "Prof-school",
        "5th-6th",
        "10th",
        "Preschool",
        "12th",
        "1st-4th",
    ]
    marital_status: Literal[
        "Never-married",
        "Married-civ-spouse",
        "Divorced",
        "Married-spouse-absent",
        "Separated",
        "Married-AF-spouse",
        "Widowed",
    ]
    occupation: Literal[
        "Adm-clerical",
        "Exec-managerial",
        "Handlers-cleaners",
        "Prof-specialty",
        "Other-service",
        "Sales",
        "Transport-moving",
        "Farming-fishing",
        "Machine-op-inspct",
        "Tech-support",
        "Craft-repair",
        "Protective-serv",
        "Armed-Forces",
        "Priv-house-serv",
    ]
    relationship: Literal[
        "Not-in-family",
        "Husband",
        "Wife",
        "Own-child",
        "Unmarried",
        "Other-relative",
    ]
    race: Literal[
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
    ]
    sex: Literal["Male", "Female"]
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: Literal[
        "United-States",
        "Cuba",
        "Jamaica",
        "India",
        "Mexico",
        "Puerto-Rico",
        "Honduras",
        "England",
        "Canada",
        "Germany",
        "Iran",
        "Philippines",
        "Poland",
        "Columbia",
        "Cambodia",
        "Thailand",
        "Ecuador",
        "Laos",
        "Taiwan",
        "Haiti",
        "Portugal",
        "Dominican-Republic",
        "El-Salvador",
        "France",
        "Guatemala",
        "Italy",
        "China",
        "South",
        "Japan",
        "Yugoslavia",
        "Peru",
        "Outlying-US(Guam-USVI-etc)",
        "Scotland",
        "Trinadad&Tobago",
        "Greece",
        "Nicaragua",
        "Vietnam",
        "Hong",
        "Ireland",
        "Hungary",
        "Holand-Netherlands",
    ]

    class Config:
        """
        POST method input example
        """

        schema_extra = {
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


# thanks to:
# https://github.com/ximenesfel
# for the hard work
# ------
# **to finish later**
# class Census(BaseModel):
#     age: float
# workclass: Literal[
#     'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
#     'Local-gov', 'Self-emp-inc', 'Without-pay']
#     education: Literal[
#         'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
#         'Some-college',
#         'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
#         '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
#     maritalStatus: Literal[
#         'Never-married', 'Married-civ-spouse', 'Divorced',
#         'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
#         'Widowed']
#     occupation: Literal[
#         'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
#         'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
#         'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
#         'Craft-repair', 'Protective-serv', 'Armed-Forces',
#         'Priv-house-serv']
#     relationship: Literal[
#         'Not-in-family', 'Husband', 'Wife', 'Own-child',
#         'Unmarried', 'Other-relative']
#     race: Literal[
#         'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
#         'Other']
#     sex: Literal['Male', 'Female']
#     hoursPerWeek: float
#     nativeCountry: Literal[
#         'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
#         'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
#         'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
#         'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
#         'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
#         'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
#         'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
#         'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
#         'Holand-Netherlands']
