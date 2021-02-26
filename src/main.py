#!/usr/bin/env python
"""
File: main.py
Author: Greg, Leon & Jonny
Github: https://github.com/headcase
Description: main file
"""

from load import load
from load import clean

def main():
    """ main function """

    drops = ['PoolQC', 'MiscFeature', 'FireplaceQu', 'Id']
    fills = {'MasVnrArea': 0.0, 'LotFrontage': 0.0}
    
    train_data = load()
    train_data = clean(train_data, drop_list=drops, fill_na=fills)
    
    # The electrical variable has one null value, we elect to delete this
    # observation
    elec_na = train_data["Electrical"].isna()
    train_data.drop(elec_na.loc[elec_na == True].index, inplace=True)

    print(train_data)

if __name__ == '__main__':
    main()


