#!/usr/bin/env python
""" Loading of housing price training data into pandas dataframe"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PowerTransformer


def clean(df, drop_list=[], fill_na={}):
    """Helper function for cleaning up Housing Prices dataframe"""
    
    # Compute operations on fresh copy of DataFrame to avoid errors in Jupyter
    # notebooks when editing source DataFrame directly
    update_df = df.copy()

    if drop_list:
        # Drop any columns supplied in the drop_list
        update_df = update_df.drop(drop_list, axis=1)

    if fill_na:
        # For any key, value pairs in the supplied dictionary, set null values
        # to fill_val for the given variable
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

    for out_feat, in_feat in feat_dict.items():
        new_col = pd.Series()

        for multiplier, feats in in_feat.items():
            new_col = new_col.add(multiplier * update_df[feats].sum(axis=1),
                    fill_value=0)

        update_df[out_feat] = new_col

    return update_df

def ordinal_create():
    pass

def preprocess(df, scale_list=[], transform_list=[], dummies=True):
    """Scales, transforms, and computes dummies for variables in training
    dataset"""

    # Compute operations on fresh copy of DataFrame to avoid errors in Jupyter
    # notebooks when editing source DataFrame directly
    update_df = df.copy()

    pipe_dict = {}

    if scale_list:
        for var in scale_list:
            if var in transform_list:
                pipeline = Pipeline([('scaler', RobustScaler()),('transform',
                    PowerTransformer())])                
            else:
                pipeline = Pipeline([('scaler', RobustScaler())])                

            update_df[var] = pipeline.fit(update_df[[var]]).transform(update_df[[var]])
            pipe_dict[var] = pipeline

    # Convert categorical variables to numerical dummy variables
    if dummies:
        update_df = pd.get_dummies(update_df)

    return update_df, pipe_dict

def null_match(df, cols_list):
    update_df = df.copy()
    for sub_list in cols_list:
        i = 0
        j = len(sub_list)
        de = deque(sub_list)  
        while i < j:
            base_col = de.pop()
            mask = update_df[base_col].isnull()
            update_df.loc[mask, sub_list] = np.nan
            de.appendleft(base_col)
            i += 1
    return update_df
