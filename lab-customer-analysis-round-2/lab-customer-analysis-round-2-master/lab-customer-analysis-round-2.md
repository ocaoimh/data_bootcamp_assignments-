```python
import pandas as pd # import library 

```


```python
# Import data 

dflab2 = pd.read_csv('marketing_customer_analysis.csv')


```


```python
dflab2
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
      <th>Unnamed: 0</th>
      <th>Customer</th>
      <th>State</th>
      <th>Customer Lifetime Value</th>
      <th>Response</th>
      <th>Coverage</th>
      <th>Education</th>
      <th>Effective To Date</th>
      <th>EmploymentStatus</th>
      <th>Gender</th>
      <th>...</th>
      <th>Number of Open Complaints</th>
      <th>Number of Policies</th>
      <th>Policy Type</th>
      <th>Policy</th>
      <th>Renew Offer Type</th>
      <th>Sales Channel</th>
      <th>Total Claim Amount</th>
      <th>Vehicle Class</th>
      <th>Vehicle Size</th>
      <th>Vehicle Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>DK49336</td>
      <td>Arizona</td>
      <td>4809.216960</td>
      <td>No</td>
      <td>Basic</td>
      <td>College</td>
      <td>2/18/11</td>
      <td>Employed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>9</td>
      <td>Corporate Auto</td>
      <td>Corporate L3</td>
      <td>Offer3</td>
      <td>Agent</td>
      <td>292.800000</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>KX64629</td>
      <td>California</td>
      <td>2228.525238</td>
      <td>No</td>
      <td>Basic</td>
      <td>College</td>
      <td>1/18/11</td>
      <td>Unemployed</td>
      <td>F</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>Personal Auto</td>
      <td>Personal L3</td>
      <td>Offer4</td>
      <td>Call Center</td>
      <td>744.924331</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>LZ68649</td>
      <td>Washington</td>
      <td>14947.917300</td>
      <td>No</td>
      <td>Basic</td>
      <td>Bachelor</td>
      <td>2/10/11</td>
      <td>Employed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>Personal Auto</td>
      <td>Personal L3</td>
      <td>Offer3</td>
      <td>Call Center</td>
      <td>480.000000</td>
      <td>SUV</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>XL78013</td>
      <td>Oregon</td>
      <td>22332.439460</td>
      <td>Yes</td>
      <td>Extended</td>
      <td>College</td>
      <td>1/11/11</td>
      <td>Employed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>Corporate Auto</td>
      <td>Corporate L3</td>
      <td>Offer2</td>
      <td>Branch</td>
      <td>484.013411</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>QA50777</td>
      <td>Oregon</td>
      <td>9025.067525</td>
      <td>No</td>
      <td>Premium</td>
      <td>Bachelor</td>
      <td>1/17/11</td>
      <td>Medical Leave</td>
      <td>F</td>
      <td>...</td>
      <td>NaN</td>
      <td>7</td>
      <td>Personal Auto</td>
      <td>Personal L2</td>
      <td>Offer1</td>
      <td>Branch</td>
      <td>707.925645</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10905</th>
      <td>10905</td>
      <td>FE99816</td>
      <td>Nevada</td>
      <td>15563.369440</td>
      <td>No</td>
      <td>Premium</td>
      <td>Bachelor</td>
      <td>1/19/11</td>
      <td>Unemployed</td>
      <td>F</td>
      <td>...</td>
      <td>NaN</td>
      <td>7</td>
      <td>Personal Auto</td>
      <td>Personal L1</td>
      <td>Offer3</td>
      <td>Web</td>
      <td>1214.400000</td>
      <td>Luxury Car</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10906</th>
      <td>10906</td>
      <td>KX53892</td>
      <td>Oregon</td>
      <td>5259.444853</td>
      <td>No</td>
      <td>Basic</td>
      <td>College</td>
      <td>1/6/11</td>
      <td>Employed</td>
      <td>F</td>
      <td>...</td>
      <td>0.0</td>
      <td>6</td>
      <td>Personal Auto</td>
      <td>Personal L3</td>
      <td>Offer2</td>
      <td>Branch</td>
      <td>273.018929</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10907</th>
      <td>10907</td>
      <td>TL39050</td>
      <td>Arizona</td>
      <td>23893.304100</td>
      <td>No</td>
      <td>Extended</td>
      <td>Bachelor</td>
      <td>2/6/11</td>
      <td>Employed</td>
      <td>F</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>Corporate Auto</td>
      <td>Corporate L3</td>
      <td>Offer1</td>
      <td>Web</td>
      <td>381.306996</td>
      <td>Luxury SUV</td>
      <td>Medsize</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10908</th>
      <td>10908</td>
      <td>WA60547</td>
      <td>California</td>
      <td>11971.977650</td>
      <td>No</td>
      <td>Premium</td>
      <td>College</td>
      <td>2/13/11</td>
      <td>Employed</td>
      <td>F</td>
      <td>...</td>
      <td>4.0</td>
      <td>6</td>
      <td>Personal Auto</td>
      <td>Personal L1</td>
      <td>Offer1</td>
      <td>Branch</td>
      <td>618.288849</td>
      <td>SUV</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10909</th>
      <td>10909</td>
      <td>IV32877</td>
      <td>NaN</td>
      <td>6857.519928</td>
      <td>NaN</td>
      <td>Basic</td>
      <td>Bachelor</td>
      <td>1/8/11</td>
      <td>Unemployed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>3</td>
      <td>Personal Auto</td>
      <td>Personal L1</td>
      <td>Offer4</td>
      <td>Web</td>
      <td>1021.719397</td>
      <td>SUV</td>
      <td>Medsize</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10910 rows × 26 columns</p>
</div>




```python
# Show the dataframe shape.
dflab2.columns


```




    Index(['Unnamed: 0', 'Customer', 'State', 'Customer Lifetime Value',
           'Response', 'Coverage', 'Education', 'Effective To Date',
           'EmploymentStatus', 'Gender', 'Income', 'Location Code',
           'Marital Status', 'Monthly Premium Auto', 'Months Since Last Claim',
           'Months Since Policy Inception', 'Number of Open Complaints',
           'Number of Policies', 'Policy Type', 'Policy', 'Renew Offer Type',
           'Sales Channel', 'Total Claim Amount', 'Vehicle Class', 'Vehicle Size',
           'Vehicle Type'],
          dtype='object')




```python
dflab2.dtypes # show data types 

```




    Unnamed: 0                         int64
    Customer                          object
    State                             object
    Customer Lifetime Value          float64
    Response                          object
    Coverage                          object
    Education                         object
    Effective To Date                 object
    EmploymentStatus                  object
    Gender                            object
    Income                             int64
    Location Code                     object
    Marital Status                    object
    Monthly Premium Auto               int64
    Months Since Last Claim          float64
    Months Since Policy Inception      int64
    Number of Open Complaints        float64
    Number of Policies                 int64
    Policy Type                       object
    Policy                            object
    Renew Offer Type                  object
    Sales Channel                     object
    Total Claim Amount               float64
    Vehicle Class                     object
    Vehicle Size                      object
    Vehicle Type                      object
    dtype: object




```python
# 2. Standardize header names.

dflab2.columns= dflab2.columns.str.lower() #make column names lower case
dflab2.columns

```




    Index(['unnamed: 0', 'customer', 'state', 'customer lifetime value',
           'response', 'coverage', 'education', 'effective to date',
           'employmentstatus', 'gender', 'income', 'location code',
           'marital status', 'monthly premium auto', 'months since last claim',
           'months since policy inception', 'number of open complaints',
           'number of policies', 'policy type', 'policy', 'renew offer type',
           'sales channel', 'total claim amount', 'vehicle class', 'vehicle size',
           'vehicle type'],
          dtype='object')




```python
dflab2.columns= [col_name.lower().replace(' ', '_') for col_name in dflab2.columns] # remove whitespace 
dflab2.columns

```




    Index(['unnamed:_0', 'customer', 'state', 'customer_lifetime_value',
           'response', 'coverage', 'education', 'effective_to_date',
           'employmentstatus', 'gender', 'income', 'location_code',
           'marital_status', 'monthly_premium_auto', 'months_since_last_claim',
           'months_since_policy_inception', 'number_of_open_complaints',
           'number_of_policies', 'policy_type', 'policy', 'renew_offer_type',
           'sales_channel', 'total_claim_amount', 'vehicle_class', 'vehicle_size',
           'vehicle_type'],
          dtype='object')




```python
# 3. Which columns are numerical?
dflab2.select_dtypes(exclude=['object']).columns.tolist()

```




    ['unnamed:_0',
     'customer_lifetime_value',
     'income',
     'monthly_premium_auto',
     'months_since_last_claim',
     'months_since_policy_inception',
     'number_of_open_complaints',
     'number_of_policies',
     'total_claim_amount']




```python
# 4. Which columns are categorical?
dflab2.select_dtypes(include=['object']).columns.tolist()

```




    ['customer',
     'state',
     'response',
     'coverage',
     'education',
     'effective_to_date',
     'employmentstatus',
     'gender',
     'location_code',
     'marital_status',
     'policy_type',
     'policy',
     'renew_offer_type',
     'sales_channel',
     'vehicle_class',
     'vehicle_size',
     'vehicle_type']




```python
# 5. Check and deal with `NaN` values.
dflab2.isna().sum() # counts the number of nans per column 


```




    unnamed:_0                          0
    customer                            0
    state                             631
    customer_lifetime_value             0
    response                          631
    coverage                            0
    education                           0
    effective_to_date                   0
    employmentstatus                    0
    gender                              0
    income                              0
    location_code                       0
    marital_status                      0
    monthly_premium_auto                0
    months_since_last_claim           633
    months_since_policy_inception       0
    number_of_open_complaints         633
    number_of_policies                  0
    policy_type                         0
    policy                              0
    renew_offer_type                    0
    sales_channel                       0
    total_claim_amount                  0
    vehicle_class                     622
    vehicle_size                      622
    vehicle_type                     5482
    dtype: int64




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




    unnamed:_0                       0
    customer                         0
    state                            0
    customer_lifetime_value          0
    response                         0
    coverage                         0
    education                        0
    effective_to_date                0
    employmentstatus                 0
    gender                           0
    income                           0
    location_code                    0
    marital_status                   0
    monthly_premium_auto             0
    months_since_last_claim          0
    months_since_policy_inception    0
    number_of_open_complaints        0
    number_of_policies               0
    policy_type                      0
    policy                           0
    renew_offer_type                 0
    sales_channel                    0
    total_claim_amount               0
    vehicle_class                    0
    vehicle_size                     0
    vehicle_type                     0
    dtype: int64




```python
# 6. Datetime format - Extract the months from the dataset and store in a separate column.
from datetime import datetime


dflab3 = dflab2
```


```python

dflab3['effective_to_date'] = pd.to_datetime(dflab3['effective_to_date'], errors = 'coerce')


```


```python

dflab3.dtypes


```




    unnamed:_0                                int64
    customer                                 object
    state                                    object
    customer_lifetime_value                 float64
    response                                 object
    coverage                                 object
    education                                object
    effective_to_date                datetime64[ns]
    employmentstatus                         object
    gender                                   object
    income                                    int64
    location_code                            object
    marital_status                           object
    monthly_premium_auto                      int64
    months_since_last_claim                  object
    months_since_policy_inception             int64
    number_of_open_complaints                object
    number_of_policies                        int64
    policy_type                              object
    policy                                   object
    renew_offer_type                         object
    sales_channel                            object
    total_claim_amount                      float64
    vehicle_class                            object
    vehicle_size                             object
    vehicle_type                             object
    dtype: object




```python

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
      <th>unnamed:_0</th>
      <th>customer</th>
      <th>state</th>
      <th>customer_lifetime_value</th>
      <th>response</th>
      <th>coverage</th>
      <th>education</th>
      <th>effective_to_date</th>
      <th>employmentstatus</th>
      <th>gender</th>
      <th>...</th>
      <th>number_of_open_complaints</th>
      <th>number_of_policies</th>
      <th>policy_type</th>
      <th>policy</th>
      <th>renew_offer_type</th>
      <th>sales_channel</th>
      <th>total_claim_amount</th>
      <th>vehicle_class</th>
      <th>vehicle_size</th>
      <th>vehicle_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>DK49336</td>
      <td>Arizona</td>
      <td>4809.216960</td>
      <td>No</td>
      <td>Basic</td>
      <td>College</td>
      <td>2011-02-18</td>
      <td>Employed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>9</td>
      <td>Corporate Auto</td>
      <td>Corporate L3</td>
      <td>Offer3</td>
      <td>Agent</td>
      <td>292.800000</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>KX64629</td>
      <td>California</td>
      <td>2228.525238</td>
      <td>No</td>
      <td>Basic</td>
      <td>College</td>
      <td>2011-01-18</td>
      <td>Unemployed</td>
      <td>F</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>Personal Auto</td>
      <td>Personal L3</td>
      <td>Offer4</td>
      <td>Call Center</td>
      <td>744.924331</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>LZ68649</td>
      <td>Washington</td>
      <td>14947.917300</td>
      <td>No</td>
      <td>Basic</td>
      <td>Bachelor</td>
      <td>2011-02-10</td>
      <td>Employed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>Personal Auto</td>
      <td>Personal L3</td>
      <td>Offer3</td>
      <td>Call Center</td>
      <td>480.000000</td>
      <td>SUV</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>XL78013</td>
      <td>Oregon</td>
      <td>22332.439460</td>
      <td>Yes</td>
      <td>Extended</td>
      <td>College</td>
      <td>2011-01-11</td>
      <td>Employed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>Corporate Auto</td>
      <td>Corporate L3</td>
      <td>Offer2</td>
      <td>Branch</td>
      <td>484.013411</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>QA50777</td>
      <td>Oregon</td>
      <td>9025.067525</td>
      <td>No</td>
      <td>Premium</td>
      <td>Bachelor</td>
      <td>2011-01-17</td>
      <td>Medical Leave</td>
      <td>F</td>
      <td>...</td>
      <td>0</td>
      <td>7</td>
      <td>Personal Auto</td>
      <td>Personal L2</td>
      <td>Offer1</td>
      <td>Branch</td>
      <td>707.925645</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10905</th>
      <td>10905</td>
      <td>FE99816</td>
      <td>Nevada</td>
      <td>15563.369440</td>
      <td>No</td>
      <td>Premium</td>
      <td>Bachelor</td>
      <td>2011-01-19</td>
      <td>Unemployed</td>
      <td>F</td>
      <td>...</td>
      <td>0</td>
      <td>7</td>
      <td>Personal Auto</td>
      <td>Personal L1</td>
      <td>Offer3</td>
      <td>Web</td>
      <td>1214.400000</td>
      <td>Luxury Car</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10906</th>
      <td>10906</td>
      <td>KX53892</td>
      <td>Oregon</td>
      <td>5259.444853</td>
      <td>No</td>
      <td>Basic</td>
      <td>College</td>
      <td>2011-01-06</td>
      <td>Employed</td>
      <td>F</td>
      <td>...</td>
      <td>0.0</td>
      <td>6</td>
      <td>Personal Auto</td>
      <td>Personal L3</td>
      <td>Offer2</td>
      <td>Branch</td>
      <td>273.018929</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10907</th>
      <td>10907</td>
      <td>TL39050</td>
      <td>Arizona</td>
      <td>23893.304100</td>
      <td>No</td>
      <td>Extended</td>
      <td>Bachelor</td>
      <td>2011-02-06</td>
      <td>Employed</td>
      <td>F</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>Corporate Auto</td>
      <td>Corporate L3</td>
      <td>Offer1</td>
      <td>Web</td>
      <td>381.306996</td>
      <td>Luxury SUV</td>
      <td>Medsize</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10908</th>
      <td>10908</td>
      <td>WA60547</td>
      <td>California</td>
      <td>11971.977650</td>
      <td>No</td>
      <td>Premium</td>
      <td>College</td>
      <td>2011-02-13</td>
      <td>Employed</td>
      <td>F</td>
      <td>...</td>
      <td>4.0</td>
      <td>6</td>
      <td>Personal Auto</td>
      <td>Personal L1</td>
      <td>Offer1</td>
      <td>Branch</td>
      <td>618.288849</td>
      <td>SUV</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10909</th>
      <td>10909</td>
      <td>IV32877</td>
      <td>0</td>
      <td>6857.519928</td>
      <td>0</td>
      <td>Basic</td>
      <td>Bachelor</td>
      <td>2011-01-08</td>
      <td>Unemployed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>3</td>
      <td>Personal Auto</td>
      <td>Personal L1</td>
      <td>Offer4</td>
      <td>Web</td>
      <td>1021.719397</td>
      <td>SUV</td>
      <td>Medsize</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10910 rows × 26 columns</p>
</div>




```python
#Then filter the data to show only the information for the first quarter , ie. January, February and March. _Hint_: If data from March does not exist, consider only January and February.
#filtered_df = dflab3.loc[dflab3['effective_to_date'].dt.month == 3]

filtered_df = dflab3.query("effective_to_date >= '2011-01-01' \
                       and effective_to_date < '2020-03-31'")

```


```python
filtered_df
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
      <th>unnamed:_0</th>
      <th>customer</th>
      <th>state</th>
      <th>customer_lifetime_value</th>
      <th>response</th>
      <th>coverage</th>
      <th>education</th>
      <th>effective_to_date</th>
      <th>employmentstatus</th>
      <th>gender</th>
      <th>...</th>
      <th>number_of_open_complaints</th>
      <th>number_of_policies</th>
      <th>policy_type</th>
      <th>policy</th>
      <th>renew_offer_type</th>
      <th>sales_channel</th>
      <th>total_claim_amount</th>
      <th>vehicle_class</th>
      <th>vehicle_size</th>
      <th>vehicle_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>DK49336</td>
      <td>Arizona</td>
      <td>4809.216960</td>
      <td>No</td>
      <td>Basic</td>
      <td>College</td>
      <td>2011-02-18</td>
      <td>Employed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>9</td>
      <td>Corporate Auto</td>
      <td>Corporate L3</td>
      <td>Offer3</td>
      <td>Agent</td>
      <td>292.800000</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>KX64629</td>
      <td>California</td>
      <td>2228.525238</td>
      <td>No</td>
      <td>Basic</td>
      <td>College</td>
      <td>2011-01-18</td>
      <td>Unemployed</td>
      <td>F</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>Personal Auto</td>
      <td>Personal L3</td>
      <td>Offer4</td>
      <td>Call Center</td>
      <td>744.924331</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>LZ68649</td>
      <td>Washington</td>
      <td>14947.917300</td>
      <td>No</td>
      <td>Basic</td>
      <td>Bachelor</td>
      <td>2011-02-10</td>
      <td>Employed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>Personal Auto</td>
      <td>Personal L3</td>
      <td>Offer3</td>
      <td>Call Center</td>
      <td>480.000000</td>
      <td>SUV</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>XL78013</td>
      <td>Oregon</td>
      <td>22332.439460</td>
      <td>Yes</td>
      <td>Extended</td>
      <td>College</td>
      <td>2011-01-11</td>
      <td>Employed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>Corporate Auto</td>
      <td>Corporate L3</td>
      <td>Offer2</td>
      <td>Branch</td>
      <td>484.013411</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>QA50777</td>
      <td>Oregon</td>
      <td>9025.067525</td>
      <td>No</td>
      <td>Premium</td>
      <td>Bachelor</td>
      <td>2011-01-17</td>
      <td>Medical Leave</td>
      <td>F</td>
      <td>...</td>
      <td>0</td>
      <td>7</td>
      <td>Personal Auto</td>
      <td>Personal L2</td>
      <td>Offer1</td>
      <td>Branch</td>
      <td>707.925645</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10905</th>
      <td>10905</td>
      <td>FE99816</td>
      <td>Nevada</td>
      <td>15563.369440</td>
      <td>No</td>
      <td>Premium</td>
      <td>Bachelor</td>
      <td>2011-01-19</td>
      <td>Unemployed</td>
      <td>F</td>
      <td>...</td>
      <td>0</td>
      <td>7</td>
      <td>Personal Auto</td>
      <td>Personal L1</td>
      <td>Offer3</td>
      <td>Web</td>
      <td>1214.400000</td>
      <td>Luxury Car</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10906</th>
      <td>10906</td>
      <td>KX53892</td>
      <td>Oregon</td>
      <td>5259.444853</td>
      <td>No</td>
      <td>Basic</td>
      <td>College</td>
      <td>2011-01-06</td>
      <td>Employed</td>
      <td>F</td>
      <td>...</td>
      <td>0.0</td>
      <td>6</td>
      <td>Personal Auto</td>
      <td>Personal L3</td>
      <td>Offer2</td>
      <td>Branch</td>
      <td>273.018929</td>
      <td>Four-Door Car</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10907</th>
      <td>10907</td>
      <td>TL39050</td>
      <td>Arizona</td>
      <td>23893.304100</td>
      <td>No</td>
      <td>Extended</td>
      <td>Bachelor</td>
      <td>2011-02-06</td>
      <td>Employed</td>
      <td>F</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>Corporate Auto</td>
      <td>Corporate L3</td>
      <td>Offer1</td>
      <td>Web</td>
      <td>381.306996</td>
      <td>Luxury SUV</td>
      <td>Medsize</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10908</th>
      <td>10908</td>
      <td>WA60547</td>
      <td>California</td>
      <td>11971.977650</td>
      <td>No</td>
      <td>Premium</td>
      <td>College</td>
      <td>2011-02-13</td>
      <td>Employed</td>
      <td>F</td>
      <td>...</td>
      <td>4.0</td>
      <td>6</td>
      <td>Personal Auto</td>
      <td>Personal L1</td>
      <td>Offer1</td>
      <td>Branch</td>
      <td>618.288849</td>
      <td>SUV</td>
      <td>Medsize</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10909</th>
      <td>10909</td>
      <td>IV32877</td>
      <td>0</td>
      <td>6857.519928</td>
      <td>0</td>
      <td>Basic</td>
      <td>Bachelor</td>
      <td>2011-01-08</td>
      <td>Unemployed</td>
      <td>M</td>
      <td>...</td>
      <td>0.0</td>
      <td>3</td>
      <td>Personal Auto</td>
      <td>Personal L1</td>
      <td>Offer4</td>
      <td>Web</td>
      <td>1021.719397</td>
      <td>SUV</td>
      <td>Medsize</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10910 rows × 26 columns</p>
</div>




```python
dflab4
```


```python
#dflab4['effective_to_date'] = pd.to_datetime(dflab4['effective_to_date'], format='%Y-%m-%d').dt.month_name().str.slice(stop=3) # convert month names to letters 

```


```python


```


```python
# 7. BONUS: Put all the previously mentioned data transformations into a function.

```


```python
#
```
