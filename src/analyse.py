#!/usr/bin/env python
"""
File: fit.py
Description: Functions for analysing the distribution of input variables,
fitting a distribution to said variables, and transforming them accordingly
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from numpy import reshape
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer


def analyse(variable):
    """Accepts a pandas Series or DataFrame and returns skewness and kurtosis
    analysis of the variable(s)"""

    # Pandas has built-in skew and kurtosis functions for DataFrames and
    # Series, so we just call those on our input variable and organise them for
    # printing
    if isinstance(variable, pd.core.frame.Series):
        results = pd.DataFrame({"Skewness": variable.skew(), "Kurtosis":
            variable.kurtosis()}, index=[variable.name])
        print(results)
    elif isinstance(variable, pd.core.frame.DataFrame):
        results = pd.DataFrame()
        results["Skewness"] = variable.skew()
        results["Kurtosis"] = variable.kurtosis()
        print(results.sort_values("Skewness", ascending=False))
    else:
        print("You must supply either a pandas Series or DataFrame. You gave{}".format(type(variable)))

def test_trans(variable):
    """Accepts data for a single variable in the form of a pandas Series and
    computes transformation of this variable using sklearn PowerTransformer and
    plots before and after results to check for normality"""

    # scaler = RobustScaler() 
    scaler = StandardScaler() 
    col_name = variable.columns[0]
    variable = variable.rename(columns={col_name:'raw'})
    scaler.fit(variable)
    variable['raw'] = scaler.transform(variable)
    fig, axs = plt.subplots(1, 2)
    pt = PowerTransformer(standardize=False)
    pt.fit(variable)
    trans_var = pt.transform(variable)
    variable['transformed'] = trans_var
    sns.histplot(variable, x='raw', ax=axs[0])
    sns.histplot(variable, x='transformed', ax=axs[1])
    plt.show()

    # return trans_var

