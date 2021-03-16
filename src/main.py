#!/usr/bin/env python
"""
File: main.py
Author: Greg, Leon & Jonny
Github: https://github.com/headcase
Description: main file
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import clean
# from preprocess import preprocess
from preprocess import feat_create
from preprocess import ordinal_create


def main():
    """ main function """

    drops = ['PoolQC', 'MiscFeature', 'FireplaceQu', 'Id'] 
    fills = {0.0: ['MasVnrArea', 'LotFrontage'], 'NA': ['ExterQual',
        'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
        'GarageQual', 'GarageCond']} 
    scale = ['SalePrice', 'LotArea']
    transform = ['SalePrice']
    ordinal_vars = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
            'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']
    new_feats = {
            "Total_Bath": 
                {
                    1:['BsmtFullBath','FullBath'], 
                    0.5: ['BsmtHalfBath', 'HalfBath']
                },
            "Porch_SF":
                {
                    1: ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                        '3SsnPorch', 'ScreenPorch']
                },
            "Total_SF":
                {
                    1: ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'] 
                }
            }
    
    train_data = pd.read_csv('../data/train.csv')

    # The electrical variable has one null value, we elect to delete this
    # observation
    elec_na = train_data["Electrical"].isna()
    clean_data = train_data.drop(elec_na.loc[elec_na == True].index)

    clean_data = clean(clean_data, drop_list=drops, fill_na=fills)
    add_feats = feat_create(clean_data, new_feats)
    ordinal_update = ordinal_create(add_feats, ordinal_vars)
    print(ordinal_update[ordinal_vars].head(20))


if __name__ == '__main__':
    main()
