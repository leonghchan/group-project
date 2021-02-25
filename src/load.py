#!/usr/bin/env python
""" Loading of housing price training data into pandas dataframe"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PowerTransformer


def load(csv='train'):
    """Read CSV file and return pandas dataframe. Default is to read training
    data, but supplying 'test' string as argument reads validation dataset"""

    df = pd.read_csv("../data/{}.csv".format(csv))

    return df


def clean(df, drop_list=[], fill_na={}, dummies=True):
    """Helper function for cleaning up Housing Prices dataframe"""


    if drop_list:
        # Electrical variable has single null value, we choose to drop it
        df.drop(1379, inplace=True)

        # Drop any columns supplied in the drop_list
        df.drop(drop_list, inplace=True, axis=1)

    if fill_na:
        # For any key, value pairs in the supplied dictionary, set null values
        # to fill_val for the given variable
        for variable, fill_val in fill_na.items():
            df[variable] = df[variable].fillna(value=fill_val)

    return df


def preprocess(df, scale_list=[], transform_list=[], dummies=True):
    """Scales, transforms, and computes dummies for variables in training
    dataset"""

    if dummies:
        # Convert categorical variables to numerical dummy variables
        df = pd.get_dummies(df)

    if scale_list:
        pass

    if transform_list:
        pass

