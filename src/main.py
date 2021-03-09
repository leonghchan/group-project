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
from analyse import test_trans


def main():
    """ main function """

    drops = ['PoolQC', 'MiscFeature', 'FireplaceQu', 'Id']
    fills = {'MasVnrArea': 0.0, 'LotFrontage': 0.0}
    scale = ['SalePrice', 'LotArea']
    transform = ['SalePrice']
    
    train_data = pd.read_csv('../data/train.csv')

    # The electrical variable has one null value, we elect to delete this
    # observation
    elec_na = train_data["Electrical"].isna()
    clean_data = train_data.drop(elec_na.loc[elec_na == True].index)

    clean_data = clean(clean_data, drop_list=drops, fill_na=fills)
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
    add_feats = feat_create(clean_data, new_feats)
    print(add_feats[['Total_Bath', 'Porch_SF', 'Total_SF']])


if __name__ == '__main__':
    main()


