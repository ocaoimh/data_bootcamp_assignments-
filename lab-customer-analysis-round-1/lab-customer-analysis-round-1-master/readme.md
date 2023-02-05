![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Customer Analysis Round 1

#### Remember the process:

1. Case Study
2. Get data
3. Cleaning/Wrangling/EDA
4. Processing Data
5. Modeling
6. Validation
7. Reporting

### Abstract

The objective of this data is to understand customer demographics and buying behavior. Later during the week, we will use predictive analytics to analyze the most profitable customers and how they interact. After that, we will take targeted actions to increase profitable customer response, retention, and growth.

For this lab, we will gather the data from 3 _csv_ files that are provided in the `files_for_lab` folder. Use that data and complete the data cleaning tasks as mentioned later in the instructions.

### Solution

- Read the three files into python as dataframes
```python

import pandas as pd # import pandas library 


```


```python
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')
df3 = pd.read_csv('file3.csv')

```

- Concatenate the three dataframes

```python
data = pd.concat([df1, df2, df3])
data.head()

```

- Show the DataFrame's shape.

```python

df1.columns
df2.columns
df3.columns
```

- Standardize header names.

```python
data2 = data # make new df based on original df 
data2.columns= data2.columns.str.lower() # make column names lower case

data2columns # check names are lower case



```

- Rearrange the columns in the dataframe as needed

```python
data3 = data2
data3
data3 = data3[['customer', 'st', 'gender', 'education', 'state', 'customer lifetime value','income', 'monthly premium auto', 'number of open complaints','policy type', 'vehicle class', 'total claim amount' ]]  # abitrary re
data3.columns 


```





- Which columns are numerical?

```python
data3.select_dtypes(exclude=['object']).columns.tolist()

print("Here's a list of the numerical columns: ",
      data3.select_dtypes(exclude=['object']).columns.tolist())
```


- Which columns are categorical?

```python

data3.select_dtypes(include=['object']).columns.tolist()


```


- Understand the meaning of all columns

```python
## Don't know what we're expected to do here

```


- Perform the data cleaning operations mentioned so far in class

```python

### See above

```


  - Delete the column education and the number of open complaints from the dataframe.

  ```python

data4 = data3.drop(columns=["education", "number of open complaints"])
data4.columns

```

  - Correct the values in the column customer lifetime value. They are given as a percent, so multiply them by 100 and change `dtype` to `numerical` type.

  ```python
data5 =data4

data5['customer lifetime value'] = data5['customer lifetime value'].fillna('0')  # replaces nan with 0
data5['customer lifetime value'].isna().sum() # counts the number of nans

data5['customer lifetime value'] = data5['customer lifetime value'].str.replace(r'%', '') # remove % sign
data5.dtypes # check if the clv col is now non-categorical 

data5['customer lifetime value'] = data5['customer lifetime value'].astype(float) # converts the strings into floats

data5['customer lifetime value'] = data5['customer lifetime value'].astype(int) # converts the floats into an integer 


data5['customer lifetime value']  = data5['customer lifetime value'] *100 (though personally it I think we should have divided it)



```


  - Check for duplicate rows in the data and remove if any.

  ```python
data6 = data5

data6[data6.index.duplicated()] # had a pb with duplicated indexes so found this snippet to count nb of duplicates 

data6 = data6.reset_index() # remove duplicate indexes in after concatenating 

```

  - Filter out the data for customers who have an income of 0 or less.

  ```python

print("Count of clients with an income less than 1 = ",
      data6[data6['income'] < 1]['income'].count())

```
