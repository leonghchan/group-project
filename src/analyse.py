#!/usr/bin/env python
"""
File: fit.py
Description: Functions for analysing the distribution of input variables,
fitting a distribution to said variables, and transforming them accordingly
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer


def analyse(data):
    """Accepts a pandas Series or DataFrame and returns skewness and kurtosis
    analysis of the data"""

    # Pandas has built-in skew and kurtosis functions for DataFrames and
    # Series, so we just call those on our input data and organise them for
    # printing
    if isinstance(data, pd.core.frame.Series):
        results = pd.DataFrame({"Skewness": data.skew(), "Kurtosis":
            data.kurtosis()}, index=[data.name])
        return results

    elif isinstance(data, pd.core.frame.DataFrame):
        results = pd.DataFrame()
        results["Skewness"] = data.skew()
        results["Kurtosis"] = data.kurtosis()
        results.sort_values("Skewness", ascending=False, inplace=True)
        return results
    else:
        print("You must supply either a pandas Series or DataFrame. You gave{}".format(type(data)))

def test_trans(series):
    """Accepts a single variable in the form of a pandas Series and computes
    a normality transformation of this data using sklearn RobustScaler and
    PowerTransformer and plots before and after results for verification"""

    scaler = RobustScaler()
    col_name = series.columns[0]
    series = series.rename(columns={col_name:'raw'})
    scaler.fit(series)
      
    series['scaled'] = scaler.transform(series)
    pt = PowerTransformer()
    pt.fit(series[['scaled']])
    trans_var = pt.transform(series[['scaled']])
    series['transformed'] = trans_var
    
    fig, axs = plt.subplots(1, 3)
    sns.histplot(series, x='raw', ax=axs[0])
    sns.histplot(series, x='scaled', ax=axs[1])
    sns.histplot(series, x='transformed', ax=axs[2])
    plt.show()
