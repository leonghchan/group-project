#!/usr/bin/env python
""" Loading of housing price training data into pandas dataframe"""

import pandas as pd


def load(csv='train'):
    """ Read CSV file and return pandas dataframe. Default is to read training
    data, but supplying 'test' string as argument reads validation dataset """

    df = pd.read_csv("../data/{}.csv".format(csv))

    return df

