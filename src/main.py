#!/usr/bin/env python
"""
File: main.py
Author: Greg, Leon & Jonny
Github: https://github.com/headcase
Description: main file
"""

from utils import load
from utils import split
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.svm import SVR as svr
from sklearn.metrics import mean_squared_error as mse

def main():
    """ main function """
    
    features = ["YearRemodAdd", "YearBuilt","TotRmsAbvGrd", "FullBath",
            "1stFlrSF", "TotalBsmtSF","GarageArea","GarageCars",
            "GrLivArea","OverallQual"]

    target = "SalePrice"

    train_data = load()
    train_X, valid_X, train_y, valid_y = split(train_data, features, target)

    # Decision tree
    model = dtr(random_state=1, max_depth=10, min_samples_leaf=10)
    model.fit(train_X, train_y)
    preds = model.predict(valid_X)
    print("Decision tree regressor MSE: {}".format(mse(valid_y, preds,
        squared=False)))

    # Support Vector Regression
    model_svr = svr()
    model_svr.fit(train_X, train_y)
    preds_svr = model_svr.predict(valid_X)
    print("Support vector regressor MSE: {}".format(mse(valid_y, preds_svr,
        squared=False)))

if __name__ == '__main__':
    main()


