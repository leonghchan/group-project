#!/usr/bin/env python
"""
File: model.py
Description: Wrappers for prediction models
"""

from sklearn.tree import DecisionTreeRegressor 

def dtree(X, y, seed=1):
    """ Accepts feature matrix (X) and target variable (y) to fit using
    DecisionTreeRegressor (sklearn). Returns decision tree model """

    raw_model = DecisionTreeRegressor(random_state=seed)
    fit_model = raw_model.fit(X,y)

    return fit_model
