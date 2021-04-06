#!/usr/bin/env python
"""
File: dataset.py
Description: Script to produce dataset ready for modelling
"""

import pandas as pd

from preprocess import (clean, feat_create, null_match, ordinal_create,
                        preprocess)


def make_dataset(dataset={}):
    """ Makes a production-ready dataset from Ames housing prices raw dataset.
    Accepts file paths to training and test data as dict, and returns
    production-ready training, testing, and target variable datasets"""

    # Declare constants up front
    missing_vals = ['n/a', 'na', '--']

    drops = ['PoolQC', 'MiscFeature', 'FireplaceQu', 'Id', 'Utilities']

    siblings = [[
        "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2"
    ], [
        "GarageFinish", "GarageYrBlt", "GarageQual", "GarageCond", "GarageType"
    ], ["MasVnrType", "MasVnrArea"]]

    ordinal_vars = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
        'KitchenQual', 'GarageQual', 'GarageCond'
    ]  # May want to add BsmtExposure

    new_feats = {
        "Total_Bath": {
            1: ['BsmtFullBath', 'FullBath'],
            0.5: ['BsmtHalfBath', 'HalfBath']
        },
        "Porch_SF": {
            1: [
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                'ScreenPorch'
            ]
        },
        "Total_SF": {
            1: ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
        }
    }

    swap_subclass = {
        20: '1story 1946+',
        30: '1story 1946-',
        40: '1story w attic',
        45: '1halfstory unfinish',
        50: '1halfstory finish',
        60: '2story 1946+',
        70: '2story 1946-',
        75: '2halfstory',
        80: 'split multi-level',
        85: 'split foyer',
        90: 'duplex',
        120: '1story PUD 1946+',
        150: '1halfstory PUD',
        160: '2story PUD 1946+',
        180: 'PUD multilevel',
        190: '2 family conv'
    }

    train = pd.read_csv(dataset['train'], na_values=missing_vals)
    test = pd.read_csv(dataset['test'], na_values=missing_vals)

    train['train'] = True
    test['train'] = False

    full_data = pd.concat([train, test]).reset_index(drop=True)
    full_data = clean(full_data, drop_list=drops)

    # One observation is missing electrical data point, we drop it
    elec_na = full_data["Electrical"].isna()

    full_data.drop(elec_na.loc[elec_na].index, inplace=True)
    full_data.reset_index(drop=True, inplace=True)

    full_data = null_match(full_data, siblings)

    ints = [
        col for col in full_data.columns if full_data.dtypes[col] == "int64"
    ]
    floats = [
        col for col in full_data.columns if full_data.dtypes[col] == "float64"
    ]
    cats = [
        col for col in full_data.columns if full_data.dtypes[col] == "object"
    ]

    fill_dict = {0: ints, 0.0: floats, "None": cats}

    full_data = clean(full_data, fill_na=fill_dict)

    full_data = feat_create(full_data, new_feats)
    full_data = ordinal_create(full_data, ordinal_vars)
    full_data['MSSubClass'] = full_data['MSSubClass'].map(swap_subclass)

    full_data.drop([523, 1298],
                   inplace=True)  # Drop two massive outliers, see above
    full_data.reset_index(drop=True, inplace=True)

    ordinal_vars.append('OverallQual')
    ordinal_vars.append('OverallCond')

    final_train = full_data.loc[(full_data.train == True), :].copy()
    final_train.drop(columns=['train'], inplace=True)
    final_train.reset_index(drop=True, inplace=True)

    final_test = full_data.loc[(full_data.train == False), :].copy()
    final_test.drop(columns=['train', 'SalePrice'], inplace=True)
    final_test.reset_index(drop=True, inplace=True)

    scale_feats = [
        col for col in final_train.columns
        if (final_train.dtypes[col] != "object") and (col not in ordinal_vars)
    ]
    trans_feats = [
        'SalePrice', 'LotArea', 'Total_SF', 'GrLivArea', 'LotFrontage',
        'GarageArea'
    ]
    final_train, _ = preprocess(final_train,
                                scale_list=scale_feats,
                                transform_list=trans_feats,
                                dummies=True)

    # correls = abs(final_train.corrwith(final_train.SalePrice))
    # low_corr = list(correls[correls < 0.1].index)
    # final_train.drop(columns=low_corr, inplace=True)

    # dummies = [
    #     col for col in final_train.columns
    #     if final_train.dtypes[col] == "uint8"
    # ]
    # counts = final_train[dummies].sum()
    # dummy_drops = list(counts[counts < 20].index)
    # final_train.drop(columns=dummy_drops, inplace=True)

    target = final_train.loc[:, 'SalePrice']
    final_train.drop(columns=['SalePrice'], inplace=True)
    final_train.reset_index(drop=True, inplace=True)

    return target, final_test, final_train
