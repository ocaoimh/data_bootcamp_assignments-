```python
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv('marketing_customer_analysis.csv', sep=",")
data.head()
```


```python
# X-y split.
y = data['Total Claim Amount'] # variable we want to know
X = data.drop(['Total Claim Amount'], axis=1) # other variables
X.head()
```


```python
#normalize the data ; scale the data
x_trans = MinMaxScaler().fit(X) # syntax sets new maximum and minimum between 0 and 1
x_minmax = x_trans.transform(X) 
print(x_minmax.shape)
```


```python

```


```python

```


```python

```
