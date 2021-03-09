# Housing Prices
This project examines the efficacy of various regression models on predicting sale prices of homes in Ames, Iowa. 

This 'readme' describes the high level processing we conducted, while referencing additional notebooks for specific details. 


```python
import pandas as pd
from src.load import clean

```

As our first step, we will load the training dataset from CSV straight into a `pandas` DataFrame


```python
train_data = pd.read_csv('data/train.csv')
```

Next, we will address some of the most egregious missing/null values up front (see [missing-values.ipynb](notebooks/missing-values.ipynb) for a thorough analysis of missing & null values). Further work to follow.  


```python
drops = ['PoolQC', 'MiscFeature', 'FireplaceQu', 'Id']
fills = {'MasVnrArea': 0.0, 'LotFrontage': 0.0}

elec_na = train_data["Electrical"].isna()
prelim_data = train_data.drop(elec_na.loc[elec_na == True].index)

prelim_data = clean(prelim_data, drop_list=drops, fill_na=fills)
```


```python

```
