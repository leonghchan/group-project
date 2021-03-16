# Housing Prices
This project examines the efficacy of various regression models on predicting sale prices of homes in Ames, Iowa. 

This 'readme' describes the high level processing we conducted, while referencing additional notebooks for specific details. 


```python
import pandas as pd
from src.preprocess import clean
from src.preprocess import null_match
```

As our first step, we will load the training and testing datasets from CSV straight into a `pandas` DataFrame. We will briefly combine these data to address null values on a single dataframe and then split them back apart. We also split off the target variable and drop it from the training data


```python
# Some additional strings to consider as null values
missing_values = ["n/a", "na", "--"]

train = pd.read_csv("data/train.csv", na_values = missing_values)
train_no_target = train.drop('SalePrice', axis=1)
test = pd.read_csv("data/test.csv", na_values = missing_values)

target = train['SalePrice']

# Set flag to discriminate between test and train
train_no_target['test_data'] = False
test['test_data'] = True

# Concatenate datasets and renumber the index
full_data = pd.concat([train_no_target, test]).reset_index(drop=True)

```

Next, we will address some of the most egregious missing/null values up front (see [missing-values.ipynb](notebooks/missing-values.ipynb) for a thorough analysis of missing & null values). 

We supply a list of variables to drop in their entirety, as they add little value to the dataset and are >90% null values. We also remove an observation with no value for the "Electrical" variable


```python
drops = ['PoolQC', 'MiscFeature', 'FireplaceQu', 'Id']

elec_na = full_data["Electrical"].isna()
cleaner_data = full_data.drop(elec_na.loc[elec_na == True].index)
# Make sure to drop the same index in the target variable
target = target.drop(elec_na.loc[elec_na == True].index)

cleaner_data = clean(cleaner_data, drop_list=drops)
```

Before filling null values, we must address some discrepancies in null values between 'sibling' columns. For example, the basement columns have slight mismatches in the number of null values, which some variables possessing a few extra null values. To rectify this, we set all basement variables to null if any sibling has a null value. Likewise for other sibling columns like those describing the garage.


```python
siblings = [
    ["BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "BsmtFinType2"],
    ["GarageFinish", "GarageYrBlt", "GarageQual", "GarageCond", "GarageType"],
    ["MasVnrType", "MasVnrArea"]   
]   

sib_match = null_match(cleaner_data, siblings)
```

Now we will fill all null values with more appropriate values. This is executed using a dictionary that maps fill values to variable data types. First we compute a list of variables pertaining to each data type, then we supply this to our `clean` function


```python
# Create lists of variables names for each data type: integer, float and categorical (objects).
ints = [col for col in sib_match.columns if sib_match.dtypes[col] == "int64"]
floats =  [col for col in sib_match.columns if sib_match.dtypes[col] == "float64"]
cats =  [col for col in sib_match.columns if sib_match.dtypes[col] == "object"]

fill_dict = {0: ints, 0.0: floats, "None": cats}

clean_data = clean(sib_match, fill_na=fill_dict)

# Let's confirm we've removed all nulls:
clean_data.isna().sum().sum()
```




    0



Excellent!


```python

```
