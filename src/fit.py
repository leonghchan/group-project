#!/usr/bin/env python
"""
File: fit.py
Description: Functions for analysing the distribution of input variables,
fitting a distribution to said variables, and transforming them accordingly
"""
import pandas as pd

def analyse(variable):
    """Accepts a pandas Series or DataFrame and returns skewness and kurtosis
    analysis of the variable(s)"""

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
        print("You must supply either a pandas Series or DataFrame. You gave {}".format(type(variable)))

def fit():
    pass

def transform():
    pass
