#!/usr/bin/env python
"""
File: preprocess.py
Description: Helper functions for data cleaning and feature engineering
"""

from collections import deque

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler


def clean(df, drop_list=[], fill_na={}):
    """Helper function for cleaning up Housing Prices dataframe. Accepts a
    DataFrame, a python list of column names to be dropped, and a dictionary of
    fill values (key) for corresponding list of column names (value). Returns a
    new DataFrame with drops and fills satisfied."""

    # Compute operations on fresh copy of DataFrame to avoid errors in Jupyter
    # notebooks when editing source DataFrame directly
    update_df = df.copy()

    if drop_list:
        # Drop any columns supplied in the drop_list
        update_df = update_df.drop(columns=drop_list)

    if fill_na:
        # For any key (fill value) and value (list of columns) pairs in the
        # supplied dictionary, set null values to fill_val for the given
        # variable(s)
        for fill_val, variables in fill_na.items():
            update_df[variables] = update_df[variables].fillna(value=fill_val)

    return update_df


def feat_create(df, feat_dict):
    """Accepts a DataFrame and a nested dictionary used to compute new features
    from existing features. Top-level key describes name of output column;
    second level key describes multiplier to be used on source features; values
    are lists of columns to be used for compositing. For example: {"TotalBath":
    {1: ['FullBath'], 0.5:['HalfBath']}}"""

    # Compute operations on fresh copy of DataFrame to avoid errors in Jupyter
    # notebooks when editing source DataFrame directly
    update_df = df.copy()

    # Unwrap nested dictionary per docstring
    for out_feat, in_feat in feat_dict.items():
        new_col = pd.Series()

        for multiplier, feats in in_feat.items():
            new_col = new_col.add(multiplier * update_df[feats].sum(axis=1),
                                  fill_value=0)

        update_df[out_feat] = new_col

    return update_df


def ordinal_create(df, var_list):
    """Helper function to convert variables encoded as categoricals which are
    actually ordinals. Uses a dictionary to remap, for example: 'NA' to 0, 'Po'
    to 1, etc. Accepts a DataFrame and a list of variables to remap. Returns a
    new DataFrame with edits applied. 

    Should be run after filling null values or columns with nulls come out as
    float instead of int"""

    # Dictionary for remapping strings to ordinal values
    qual_mapper = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    # Compute operations on fresh copy of DataFrame to avoid errors in Jupyter
    # notebooks when editing source DataFrame directly
    update_df = df.copy()

    for variable in var_list:
        update_df[variable] = update_df[variable].replace(qual_mapper)

    return update_df


def preprocess(df, scale_list=[], transform_list=[], dummies=True):
    """Scales, transforms, and computes dummy variables. Accepts a DataFrame,
    list of columns to scale, list of columns to transform toward normality,
    and a boolean flag for computing dummy variables using pandas built-in
    function. Returns a new DataFrame with changes applied. 

    Also returns a dictionary of variables (key) and sklearn pipelines (value)
    to record modifications to training variables such that they might be
    replicated on test data"""

    # Compute operations on fresh copy of DataFrame to avoid errors in Jupyter
    # notebooks when editing source DataFrame directly
    update_df = df.copy()
    variables = set(scale_list + transform_list)

    # dictionary to store fitted sklearn pipeline such that the same parameters
    # can later be applied to test data
    pipe_dict = {}

    # if scale_list:
    #     for var in scale_list:
    #         # If variable is flagged for both scaling and transformation,
    #         # build such a pipeline
    #         if var in transform_list:
    #             pipeline = Pipeline([('scaler', RobustScaler()),
    #                                  ('transform', PowerTransformer())])
    #         # Otherwise, just build a scaling pipeline
    #         else:
    #             pipeline = Pipeline([('scaler', RobustScaler())])

    #         update_df[var] = pipeline.fit(update_df[[var]]).transform(
    #             update_df[[var]])  # Is this line doing too much?
    #         pipe_dict[var] = pipeline

    for var in variables:
        # If variable is flagged for both scaling and transformation, build
        # such a pipeline
        if var in (transform_list and scale_list):
            pipeline = Pipeline([('scaler', RobustScaler()),
                                 ('transform', PowerTransformer())])
        # Otherwise, just build a scaling pipeline
        elif var in (scale_list):
            pipeline = Pipeline([('scaler', RobustScaler())])
        elif var in (transform_list):
            pipeline = Pipeline([('transform', PowerTransformer())])

        update_df[var] = pipeline.fit(update_df[[var]]).transform(
            update_df[[var]])  # Is this line doing too much?
        pipe_dict[var] = pipeline

    # Convert categorical variables to numerical dummy variables
    if dummies:
        update_df = pd.get_dummies(update_df)

    return update_df, pipe_dict


def null_match(df, cols_list):
    """Helper function to resolve null value discrepancies between 'sibling'
    columns. For example, most Garage variables have the same number of nulls,
    save for one or two variables, which have fewer. This function accepts a
    DataFrame and a nested list of 'sibling' columns to be matched. 

    Example input list: [["GarageType", "GarageCond", "GarageFinish"],
    ["BsmtQual", "BsmtCond"]]"""

    # Compute operations on fresh copy of DataFrame to avoid errors in Jupyter
    # notebooks when editing source DataFrame directly
    update_df = df.copy()

    # Burrow into nested list and use deque to cycle through each column in the
    # sub_list in turn, extract the rows with null values for that 'base_col'
    # and set to null all corresponding rows in the other 'sibling' columns
    for sub_list in cols_list:
        i = 0
        j = len(sub_list)
        queue = deque(sub_list)

        while i < j:
            base_col = queue.pop()
            mask = update_df[base_col].isnull()
            update_df.loc[mask, sub_list] = np.nan
            queue.appendleft(base_col)
            i += 1

    return update_df
