## Predicting house prices

# By Greg Headley and Leon Chan

This project examines the efficacy of various regression models on predicting sale prices of homes in Ames, Iowa.

This 'README' describes the high level processing we conducted, while referencing additional notebooks for specific details.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt


pd.options.display.max_rows = 1000
pd.options.display.max_columns = 200
plt.rcParams['figure.figsize'] = [10, 4]
plt.rcParams['figure.dpi'] = 100
```

## Import data, extract target, merge test & train

As our first step, we will load the training and testing datasets from CSV straight into a pandas DataFrame. We will briefly combine these data to address null values on a single dataframe and then split them back apart. We also split off the target variable and drop it from the training data.


```python
missing_values = ["n/a", "na", "--"]

train = pd.read_csv("data/train.csv", na_values = missing_values)
test = pd.read_csv("data/test.csv", na_values = missing_values)

# Set flag to discriminate between test and train
train['test_data'] = False
test['test_data'] = True

# Concatenate datasets and renumber the index
full_data = pd.concat([train, test]).reset_index(drop=True)
```


```python
full_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
      <th>test_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>196.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>150.0</td>
      <td>856.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>2.0</td>
      <td>548.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>284.0</td>
      <td>1262.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>2.0</td>
      <td>460.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>162.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>434.0</td>
      <td>920.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>2.0</td>
      <td>608.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>540.0</td>
      <td>756.0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>3.0</td>
      <td>642.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>350.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>490.0</td>
      <td>1145.0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>3.0</td>
      <td>836.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Drop useless columns, bad row

Next, we will address some of the most egregious missing/null values up front.

We supply a list of variables to drop in their entirety, as they add little value to the dataset and are >90% null values. We also remove an observation with no value for the "Electrical" variable.


```python
from src.preprocess import clean

drops = ['PoolQC', 'MiscFeature', 'FireplaceQu', 'Id', 'Utilities']

elec_na = full_data["Electrical"].isna()
full_data.drop(elec_na.loc[elec_na].index, inplace=True)

full_data = clean(full_data, drop_list=drops)
```

## Match null count of sibling columns

Before filling null values, we must address some discrepancies in null values between 'sibling' columns. For example, the basement columns have slight mismatches in the number of null values, which some variables possessing a few extra null values. To rectify this, we set all basement variables to null if any sibling has a null value. Likewise for other sibling columns like those describing the garage.


```python
# Import custom function to handle sibling columns. 
from src.preprocess import null_match

siblings = [
    ["BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2"],
    ["GarageFinish", "GarageYrBlt", "GarageQual", "GarageCond", "GarageType"],
    ["MasVnrType", "MasVnrArea"]   
]   

full_data = null_match(full_data, siblings) 
```

## Fill null values

Now we will fill all null values with more appropriate values. This is executed using a dictionary that maps fill values to variable data types. First we compute a list of variables pertaining to each data type, then we supply this to our `clean` function.


```python
# Create lists of variables names for each data type: integer, float and categorical (objects)
ints = [col for col in full_data.columns if full_data.dtypes[col] == "int64"]
floats =  [col for col in full_data.columns if full_data.dtypes[col] == "float64"]
cats =  [col for col in full_data.columns if full_data.dtypes[col] == "object"]

fill_dict = {0: ints, 0.0: floats, "None": cats}

full_data = clean(full_data, fill_na=fill_dict)

# Let's confirm we've removed all nulls:
full_data.isna().sum().sum()
```




    0



## Feature Engineering

Now, we create some interesting features consisting of various existing features in our dataset that could potentially assist us in making better predictions. In addition, we make a distinction between categorical and ordinal variables and convert variables accordingly. This is achieved by using our custom functions `feat_create` and `ordinal_create`.


```python
from src.preprocess import feat_create
from src.preprocess import ordinal_create

ordinal_vars = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
        'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond'] # May want to add BsmtExposure

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

swap_subclass = {20:'1story 1946+', 
                 30:'1story 1946-', 
                 40:'1story w attic', 
                 45:'1halfstory unfinish', 
                 50:'1halfstory finish', 
                 60:'2story 1946+', 
                 70:'2story 1946-', 
                 75:'2halfstory', 
                 80:'split multi-level', 
                 85:'split foyer', 
                 90:'duplex', 
                 120:'1story PUD 1946+', 
                 150:'1halfstory PUD', 
                 160:'2story PUD 1946+', 
                 180:'PUD multilevel', 
                 190:'2 family conv'}

full_data = feat_create(full_data, new_feats)
full_data = ordinal_create(full_data, ordinal_vars)
full_data['MSSubClass'] = full_data['MSSubClass'].map(swap_subclass)
```


```python
# Returns new dataframe with categorical variables converted to dummy variables. 
full_data = pd.get_dummies(full_data, drop_first=True)

# Split dataset into training and testing dataset. 
final_train = full_data.loc[(full_data.test_data == False), :].copy()
final_train.drop(columns=['test_data'], inplace = True)
final_train.reset_index(drop=True, inplace=True)

final_test = full_data.loc[(full_data.test_data == True), :].copy()
final_test.drop(columns=['test_data', 'SalePrice'], inplace = True)
final_test.reset_index(drop=True, inplace=True)
```

## Scale (standardise) and Transform (normalise) numeric variables

There were some numeric variables which required scaling and transforming according to our [analysis](greg-eda.ipynb). We achieved this using the custom function `preprocess` which accepts lists consisting of variables which require scaling, transforming or both. 

In addition to the dataframe, the `preprocess` function returns a dictionary of pipelines which stores the transformation objects for each variable.  


```python
from src.preprocess import preprocess

ordinal_vars.append('OverallQual')
ordinal_vars.append('OverallCond')

scale_feats =  [col for col in final_train.columns if (final_train.dtypes[col] != "object") and (col not in ordinal_vars)]
trans_feats = ['SalePrice', 'LotArea', 'Total_SF', 'GrLivArea', 'LotFrontage', 'GarageArea']

# Drop two massive outliers identified in analysis. 
final_train = final_train.drop([523, 1298]) 

final_train, pipelines = preprocess(final_train, scale_list=scale_feats, transform_list=trans_feats)
```

Before modelling, we dropped off the target from the training dataset. 


```python
target = final_train.loc[:, 'SalePrice']
final_train.drop(columns=['SalePrice'], inplace=True)
final_train.reset_index(drop=True, inplace=True)
```

## Modelling

We will try a handful of cutting edge regression models to make predictions and make comparisons between the models. 


```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge

RANDOM_SEED = 42

forest = RandomForestRegressor(n_jobs=-1, 
                               random_state=RANDOM_SEED, 
                               n_estimators=200, 
                               max_features=50, 
                               min_samples_leaf=2, 
                               min_samples_split=2, 
                               max_depth=20)
forest.fit(final_train, target)

gboost = GradientBoostingRegressor(n_estimators=1500, 
                                   learning_rate=0.03, 
                                   max_features=40, 
                                   min_samples_leaf=2, 
                                   min_samples_split=12, 
                                   random_state=RANDOM_SEED)
gboost.fit(final_train, target)

lasso = Lasso(alpha=0.0005,
              max_iter=5000,
              random_state=RANDOM_SEED)
lasso.fit(final_train, target)

ridge = Ridge(alpha=7.5,
              random_state=RANDOM_SEED)
ridge.fit(final_train, target)


def cv_rmse(model):
    rmse = -cross_val_score(model, final_train, target,
                            scoring="neg_root_mean_squared_error",
                            cv=10, n_jobs=-1)
    return (rmse)
```


```python
fscore = cv_rmse(forest)
gscore = cv_rmse(gboost)
lscore = cv_rmse(lasso)
rscore = cv_rmse(ridge)
print("RandomForest CV score is:   {:.4f} ({:.4f})".format(fscore.mean(), fscore.std()))
print("Gradient Boost CV score is: {:.4f} ({:.4f})".format(gscore.mean(), gscore.std()))
print("Lasso CV score is:          {:.4f} ({:.4f})".format(lscore.mean(), lscore.std()))
print("Ridge CV score is:          {:.4f} ({:.4f})".format(rscore.mean(), rscore.std()))
```

    RandomForest CV score is:   0.3238 (0.0352)
    Gradient Boost CV score is: 0.2768 (0.0400)
    Lasso CV score is:          0.2688 (0.0379)
    Ridge CV score is:          0.2697 (0.0358)
    


```python
# from sklearn.ensemble import StackingRegressor

# stack = StackingRegressor(
#     estimators=[
#         ('forest', forest),
#         ('gboost', gboost),
#         ('lasso', lasso),
#         ('ridge', ridge)
#     ], 
#     cv=10,
#     n_jobs=-1
# )
# stack.fit(final_train, target)

# sscore = cv_rmse(stack)
# print("Stacking CV score is: {:.4f} ({:.4f})".format(rscore.mean(), rscore.std()))
```


```python
from src.preprocess import pipe_apply

pipe_test = pipe_apply(final_test, pipelines, direction='forward')
```


```python
pipe_test['SalePrice'] = lasso.predict(pipe_test)
submission = pipe_apply(pipe_test, pipelines, direction='inverse')
```


```python
np.floor(submission.SalePrice)
```




    0       118084.0
    1       157036.0
    2       182565.0
    3       199980.0
    4       196272.0
              ...   
    1454     87560.0
    1455     78566.0
    1456    166047.0
    1457    121373.0
    1458    218004.0
    Name: SalePrice, Length: 1459, dtype: float64




```python
sub_df = pd.DataFrame()
sub_df['Id'] = test.Id 
sub_df['SalePrice'] = np.floor(submission.SalePrice)
```


```python
sub_df.to_csv('submission.csv', index=False)
```
