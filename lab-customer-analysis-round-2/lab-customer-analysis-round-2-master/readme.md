![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Customer Analysis Round 2

For this lab, we will be using the `marketing_customer_analysis.csv` file that you can find in the `files_for_lab` folder. Check out the `files_for_lab/about.md` to get more information if you are using the Online Excel.

**Note**: For the next labs we will be using the same data file. Please save the code, so that you can re-use it later in the labs following this lab.


## Tasks & solutions 

### Dealing with the data
```python
import pandas as pd # import library 

```
```python
# Import data 

dflab2 = pd.read_csv('marketing_customer_analysis.csv')


```


1. Show the dataframe shape.

```python
# Show the dataframe shape.
dflab2.columns


```
```python
dflab2.dtypes # show data types 

```


2. Standardize header names.

```python
dflab2.columns= [col_name.lower().replace(' ', '_') for col_name in dflab2.columns] # remove whitespace 
dflab2.columns

```


3. Which columns are numerical?

```python
# 3. Which columns are numerical?
dflab2.select_dtypes(exclude=['object']).columns.tolist()

```


4. Which columns are categorical?

```python
# 4. Which columns are categorical?
dflab2.select_dtypes(include=['object']).columns.tolist()

```


5. Check and deal with `NaN` values.

```python
# 5. Check and deal with `NaN` values.
dflab2.isna().sum() # counts the number of nans per column 


```

```python
dflab2['state'] = dflab2['state'].fillna('0')  # replaces nan with 0
dflab2['response'] = dflab2['response'].fillna('0')  # replaces nan with 0
dflab2['months_since_last_claim'] = dflab2['months_since_last_claim'].fillna('0')  # replaces nan with 0
dflab2['number_of_open_complaints'] = dflab2['number_of_open_complaints'].fillna('0')  # replaces nan with 0
dflab2['vehicle_class'] = dflab2['vehicle_class'].fillna('0')  # replaces nan with 0
dflab2['vehicle_size'] = dflab2['vehicle_size'].fillna('0')  # replaces nan with 0
dflab2['vehicle_type'] = dflab2['vehicle_type'].fillna('0')  # replaces nan with 0





```

```python
dflab2.isna().sum() # counts the number of nans per column 

```

6. Datetime format - Extract the months from the dataset and store in a separate column.

```python
# 6. Datetime format - Extract the months from the dataset and store in a separate column.
from datetime import datetime


dflab3 = dflab2
```


dflab3['effective_to_date_2'] = pd.to_datetime(dflab3['effective_to_date'], errors = 'coerce')


```
```python

dflab3.dtypes


```

7. Then filter the data to show only the information for the first quarter , ie. January, February and March. _Hint_: If data from March does not exist, consider only January and February.

```python
#Then filter the data to show only the information for the first quarter , ie. January, February and March. _Hint_: If data from March does not exist, consider only January and February.
#filtered_df = dflab3.loc[dflab3['effective_to_date'].dt.month == 3]

dflab4_q1_2011 = dflab3.query("effective_to_date_2 >= '2011-01-01' \
                       and effective_to_date_2 < '2020-03-31'")

```


8. BONUS: Put all the previously mentioned data transformations into a function.

