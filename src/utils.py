#!/usr/bin/env python
"""
File: utils.py
Description: Loading and preprocessing of dataset for prediction
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load(dataset="../data/train.csv"):
    """ Load the supplied file path into a pandas dataframe and return it """
    df = pd.read_csv(dataset)

    return df


def split(df, features=None, target=None, test_size=0.8, seed=1):
    """ Splits supplied dataframe, features (at least two) and target variable
    into training and validation datasets. Optionally supply a seed and/or a
    float from 0.0 to 1.0 representing size of validation set; default is 0.2
    (20%). Returns four series/dataframes: X and y for training and for
    validation """

    if features and target:
        X = df[features]
        y = df[target]
        train_X, valid_X, train_y, valid_y = train_test_split(X, y,
                test_size=test_size, random_state=1)
        return train_X, valid_X, train_y, valid_y
    else:
        print("You have not supplied enough features or target")

        

