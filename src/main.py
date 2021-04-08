#!/usr/bin/env python
"""
File: main.py
Author: Greg, Leon & Jonny
Github: https://github.com/headcase
Description: main file
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import sqrt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from dataset import make_dataset

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 200


def main():
    """ main function """

    RANDOM_SEED = 42
    paths = {'train': '../data/train.csv', 'test': '../data/test.csv'}
    target, test, train = make_dataset(paths)
    X, X_valid, y, y_valid = train_test_split(train,
                                              target,
                                              test_size=0.2,
                                              random_state=RANDOM_SEED)

    forest = RandomForestRegressor(n_jobs=-1,
                                   n_estimators=200,
                                   random_state=RANDOM_SEED,
                                   max_features=50,
                                   min_samples_leaf=2,
                                   min_samples_split=2,
                                   max_depth=20)
    forest.fit(X, y)
    predictions = forest.predict(X_valid)

    # gboost = GradientBoostingRegressor(n_estimators=1500,
    #                                    learning_rate=0.03,
    #                                    max_features=40,
    #                                    min_samples_leaf=2,
    #                                    min_samples_split=12,
    #                                    random_state=RANDOM_SEED)
    # gboost.fit(train, target)

    # lasso = Lasso(alpha=0.0005, max_iter=5000, random_state=RANDOM_SEED)
    # lasso.fit(X, y)
    # predictions = lasso.predict(X_valid)

    # ridge = Ridge(alpha=7.5, random_state=RANDOM_SEED)
    # ridge.fit(train, target)

    def cv_rmse(model):
        rmse = sqrt(-cross_val_score(model,
                                     train,
                                     target,
                                     scoring="neg_mean_squared_error",
                                     cv=5,
                                     n_jobs=-1))
        return (rmse)

    rf_score = cv_rmse(forest)
    # las_score = cv_rmse(lasso)
    # rid_score = cv_rmse(ridge)
    # gb_score = cv_rmse(gboost)

    print("RandomForest CV score is:      {:.4f} ({:.4f})".format(
        rf_score.mean(), rf_score.std()))
    # print("Gradient Boosting CV score is: {:.4f} ({:.4f})".format(
    #     gb_score.mean(), gb_score.std()))
    # print("Lasso CV score is:             {:.4f} ({:.4f})".format(
    # las_score.mean(), las_score.std()))
    # print("Ridge CV score is:             {:.4f} ({:.4f})".format(
    #     rid_score.mean(), rid_score.std()))

    # sns.scatterplot(x=y_valid, y=predictions)
    # plt.show()

    print("RandomForest score is: {:.4f}".format(
        sqrt(mean_squared_error(y_valid, predictions)).mean()))
    print("RandomForest R2 score is : {:.4f}".format(
        forest.score(X_valid, y_valid)))


if __name__ == '__main__':
    main()
