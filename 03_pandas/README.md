![pandas](./images/pandas.png)

>Pandas is a tool or package that helps you deal with the datasets (in csv, json, excel, ...), you can quickly see what are the columns, what data types are stored, and what you can do with it by cleaning and formatting that it can be ready to use later for the machine-learning algorithms. Pandas are built on top of numpy.

>The reason to use pandas can be see below where we compare the numpy with pandas. Each column of pandas is an array (pandas call it Series object). The convinence of indexes and column names makes selecting elements more straightforward in pandas than in numpy. Instead of using numbers (need to look up about it), we can use strings/labels to look up or select elements, which makes it so much easier for this task. We can even use indexes to merge different datasets.

![numpy vs pandas](./images/numpy_vs_pandas.png)

# :panda_face: pandas notes :blue_book:

## install pandas using pip
```bash
pip install pandas
```

## Create DataFrame (df) from Numpy:
```python
# create your first dataframe (df)
df = pd.DataFrame(
    np.array([
        [1,2],
        [3,4]
    ]),
    columns=['col1', 'col2']
)
print(df)
>>>    col1  col2
    0     1     2
    1     3     4

```

## Create Pandas Series from python list:
```python
import pandas as pd

pandas_series = pd.Series(
    [1, 2, 3, 4],  # Python list
    index=["a", "b", "c", "d"]  # Python list
)
print(pandas_series)
>>> a    1
    b    2
    c    3
    d    4
    dtype: int64

```

>Let's see the difference between Series and DataFrame.

![series vs df](./images/seris_vs_df.png)

## Load the `csv` File into df:
```python
import pandas as pd                        # standard to use pd alias
import numpy as np                         # guaranteed to be needed
pd.set_option('display.max_columns',85)    # to see 85 columns
pd.set_option('display.max_columns', None) # to display all colns (whatever)
pd.set_option('display.max_rows',85)       # to see 85 rows
# Don't show numbers as scientific notation
pd.set_option("display.float_format", "{:.2f}".format)

df = pd.read_csv('/path/to/data/file', index_col='col_name')
```

> See [here](https://pandas.pydata.org/docs/user_guide/io.html) for docs to see what different kinds of files pandas can load as dataframe, and also to see all the functionalities for each kinds of loads of the files.

## Load the *pickle* and *parquet* files into df:
```python
# Read a Python Pickle file
df_list = pd.read_pickle("listings_content.pkl")

# Read a PyArrow Parquet file
df_cal = pd.read_parquet("calendar_content.parquet", engine="pyarrow")
```

## inspecting the df
```python
df.head()      # prints first 5 rows of df
df.head(10)    # first 10 rows
df.tail()      # prints last 5 rows of df
df.dtypes      # returns the data types of all cols in df
df.shape       # shows the (rows,columns) numbers in tuple form
df.columns     # gives list of all columns names
df.columns.tolist()    # get the col names and make a list
df.index       # to see the names of index
df.info()      # shows more info, including dtypes of each column
df.describe()  # quick statistical overview of numerical cols
df.to_string() # render a df to a console-freindly tabular output
df.values      # gives us data only w/o index and cols, like numpy matrix

# copying df, and then working on copy
df_copy = df.copy()
```

## Selecting *Only* One Column:
```python
df['col_name']    # column name is string type (Object)
                  # preferred way
                  # returns Sereis Object
df[['col_name']]  # double brackets returns df Object
df.col_name       # works if col_name has no spaces

# also using loc or iloc
df.loc[:, 'col_name']
```
>Slicing with `loc` is inclusive, while slicing with `iloc` is exclusive for the end.

## Selecting Multiple Columns:
```python
df[['col1','col2','col3']]  # pass list of columns

# using loc:
df.loc[:, ['col1', 'col2', 'col3']]

# using loc slicing:
df.loc[:, 'col1':'col3']

# using iloc:
df.iloc[:, [0,3,5]] # all rows, specified cols

# using iloc slicing:
df.iloc[:, 0:5]     # all rows, first 5 cols
```
>Generally, if there are few typos that doesn't have any pattern in them, then it is better to access those values, and manually update them. But, if there are lots of typos and errors that have some pattern in them, then we can even use powerful methods like using `regex` to deal with it. Read [here](https://kanoki.org/2019/11/12/how-to-use-regex-in-pandas/) for more details on *regex in pandas*.

## access row(s) using `.loc[]` or `.iloc[]`
`iloc[]` is integer indexer. It needs integer to index rows. `loc[]` is more versatile in that it can acess rows based on labels, and that is easier most of the time. Nonetheless, both are important.
```python
# iloc search by integers
df.iloc[0]                     # gives first row; return Series Object
df.iloc[0:5, 0:5]              # gives 5 rows & 5 columns; returns df
df.iloc[[0,1,5,20],[2,4,9]]    # access specfic rows and columns

# loc search by labels
df.loc['row1']                            # if you want first row as above; return Series
df.loc['row1':'row5']                     # first 5 rows
df.loc['row1':'row5', 'col1':'col5']      # return first 5 rows and 5 columns
df.loc[['row1','row4'],['col1','col6']]   # return specified rows & cols
```

## methods for a single column
```python
df['col1'].value_counts(normalize=True)   # counts the number of unique values
                                          # best for categorical data types
df['col'].unique()                        # returns list of unique values in col
df['col'].replace('old_value','new_value', inplace=True)
df['col'].fillna('fill_value', inplace=True)
```

## work with index
```python
df.set_index('col_name', inplace=True)         # set col_name as index of df
                                               # col_name has to be in df
df.reset_index(inplace=True)                   # removes index and keep as col
                                               # default to 0,1,2,... index
df.sort_index(inplace=True)                    # sorts index ascending by default
df.sort_index(ascending=False, inplace=True)   # sorts index descending
```

## filtering using conditionals - filter rows and columns
```python
df[df['col_1']=='some_value']                      # filter df such that we only see where col_1==some_value
df.loc[df['col_1']=='some_value']                  # same as above, but using .loc
df.loc[df['col_1']=='some_value', ['col2','col3']] # .loc has little advantages

## using multiple conditions using & and |
df[(df['col_1']=="some_value") & (df['col_3']=="some_other_value")]
df[(df['col_1']=="some_value") | (df['col_3']=="some_other_value")]

## using loc with multiple filter condition
df.loc[(df['col_1']=="some_value") & (df['col_3']=="some_other_value"), ['col2','col5']]
df.loc[(df['col_1']=="some_value") | (df['col_3']=="some_other_value"), ['col2','col5']]

## filter using .isin()
# make a list of filter values that are in a column of interest
list_of_values = ['value1', 'value2', 'value3', 'value4', 'value5']
filt = df['col_of_interest'].isin(list_of_values)    # first create filter
df.loc[filt, ['col1', 'col2', 'col4']]               # apply filter to df

## filter using string method
# let's say your col values are strings, then you can use string method on them to filter
filt = df['col'].str.contains('Python', na=False)  # if that col value contains Python
df.loc[filt, ['col1', 'col2', 'col3']]
```
**Tips**: If you want all the cases where the filter didn't match you can just put tilde `~` sign infront of the filter.

## modifying/updating df as need
```python
# replace all spaces in col names with underscore
df.columns = df.columns.str.replacee(" ", "_")

# rename only columns we want to
df.rename(columns={'old_col1_name':'new_col1_name',
                  'old_col2_name':'new_col2_name',
                  'old_col3_name':'new_col3_name'}, inplace=True)

# update data in rows,col
df.loc['row_label', ['col1','col2']] = ['value1', 'value2']   # for single row
```

### update df using `apply`
```python
df['email'].apply(len)    # gives the length of each value in email column

# apply user defined function to apply
def update_email(email):
    return email.upper()

df['email'].apply(update_email)

# using lambda fun
df['email'].apply(lambda x: x.lower())  # changes all values to lower case

# apply to df applies to Series
df.apply(lambda x: x.min())    # here x variable will be Series Object
```

### update using `applymap`
Using `apply` on Series applies to each values in Series. Similarly, using `apply` on whole dataframe applies to each column i.e. Series objects. But, if we want to use `apply` on each values in a whole df, then we need to use `applymap`.
```python
df.applymap(len)   # here len function is applied to each values in df
```

### update using `map`
`map` method works only on Series. It is used to substitue values with another values in a Series.
```python
df['col'].map({'old_val1':'new_val1', 'old_val2':'new_val2'})
```

## Add/Remove Rows and Columns
```python
## cols:
## Adding columns
df['new_col'] = df['old_col'] * 1.10             # if old_col from df is a numeric type

# split full name to first and last and add to df
df[['first','last']] = df['full_name'].str.split(" ", expand=True)

## removing col
df.drop(columns=['col1','col2'], inplace=True)

## rows:
# add a single row
df.append({'col':'value'}, ignore_index=True)  # not very useful, only changes few col values that we specify

# append another df2 to df
df.append(df2, ignore_index=True)    # finds matching col names to put the values

# concatenate df
pd.concat([df1, df2], axis=1)       # concatenate by column

# remove rows
df.drop(index=2)    # drops index 2

# drop using conditions
df.drop(index=df[df['col']=='value'].index)  # the last index is because we need index from df to drop
```

## Sorting data
```python
df.sort_values(by='col_name', ascending=False)  # sorted by col_name
df.sort_values(by=['col1','col2'])              # if col1 has duplicates, it then looks col2 to sort

# diff col has diff way of sorting
df.sort_values(by=['col1','col2'], ascending=[False, True], inplace=True)

# sort by index
df.sort_index()           # see df sorted by index
df['col'].sort_values()   # returns Series sorted by col

# nlargest & nsmallest
df['col'].nlargest(10)    # displays 10 largest value in col
df.nlargest(10, 'col')    # returns df with 10 largest in col
df.nsmallest(10, 'col')   # same deal, now smallest
```

## Grouping and Aggregating
```python
df['col'].median()      # returns median value from col

## median by country -- group
# groupby does --> split, apply func, combine result
country_grp = df.groupby(['Country'])  # this groups df by Country
country_grp.get_group('United States') # get just one group

# apply func
df.groupby('Country')['col'].value_counts()  # each country's specific col value counts
df.groupby('Country')['Salary'].median()     # for each country, select salary col, then calculate median

# apply method need for Series Object
# for each country, look into LanguageWorkedWith col, and calculate total nb if
# it contains Python
df.groupby('Country')["LanguageWorkedWith"].apply(lambda x: x.str.contains('Python').sum())

# multiple agg func
# In addition to median, also calculate mean
df.groupby('Country')['Salary'].agg(['median', 'mean'])

## median to df
df.median()   # calculates median of all numeric col
df.describe()  # more stats of numeric cols
```

## Cleaning Data
```python
## if we just want to remove missing value
df.dropna(axis='index', how='any')   # default arguments; it removes np.nan, None

# drop if value in one of col is missing
df.dropna(axis=0, how='any', subset=['col'])

# if df has custom missing values, replace it with np.nan
df.replace('NA', np.nan, inplace=True)

df.isna()   # returns mask/filter whether values are nan

df.fillna('MISSING')  # fills np.nan with MISSING
df.fillna(0)          # fills np.nan with zero

df['col'].fillna(0, inplace=True)  # fill col with 0
col_mean = df['col'].mean()
df['col'].fillna(col_mean, inplace=True)  # fill with mean value of col

## cast dtype of col
df['col'].astype(int)     # convert col to int dtype
df['col'].astype(float)   # convert col to float dtype

## replace missing values while reading csv
# first create what are missing values inside df are
na_vals = ['NA', 'Missing', 'NAN', 'NONE', 'MISSING']
df = pd.read_csv('path/to/csv', index_col='col1', na_values=na_vals)
```

## Dates and Time Series data
```python
# convert string of col to datetime obj
df['col'] = pd.to_datetime(df['col'])    # convert col to datetime obj, if compatible

# explicitly telling how our col is formatted as datetime in original string
df['col'] = pd.to_datetime(df['col'], format='%Y-%m-%d %I-%p')

# running methods now to converted datetime object
df['date_col'].dt.day_name()    # returns the day of the week of each dates

# to see earliest date
df['date_col'].min()

# to see latest date
df['date_col'].max()

# to do time-delta (time betweent two times)
df['date_col'].max() - df['date_col'].min()

# filters by date, let's say we want to view just 2020
filt = (df['date_col'].dt.year >= '2020') & (df['date_col'].dt.year < '2022')
df.loc[filt]

# not just year but all some datetime obj
filt = (df['date_col'] >= pd.to_datetime('2019-01-01')) & (df['date_col'] < pd.to_datetime('2020-01-01'))
df.loc[filt]

# good idea to set index to datetime obj
df.set_index('date_col', inplace=True)

# slice df with help of index of datetime obj, now
df.loc['2020-01':'2020-02']

# resample df using other col, but using dateobject col as well
df['other_col'].resample('D').max()  # resample by Day and give max for each group

# resample using multiple columns
df.resample('D').mean()  # gives mean value for each col, on Daily basis
df.resample('W').agg({'col1':'mean', 'col2':'max', 'col3':'min', 'col4':'sum'})
```
**Tips**: We can also load the csv with datetime object formatted during the load-time only. See if you like that way also.

## Reading/Writing Data - other sources like excel, json, etc.
```python
# csv file
pd.read_csv('path/to/file')               # many helpful arguments you need to learn
new_df.to_csv('new_name.csv')             # to write your modified df into csv file
new_df.to_csv('new_name.tsv', sep='\t')   # tab separated csv

# excel
# first install packages
pip install xlwt openpyxl xlrd
# to write to excel
new_df.to_excel('new_name.xlsx')
# to specific sheet also you can do
# to read excel file
xcel_df = pd.read_excel('new_name.xlsx', index_col='col')

# json
new_df.to_json('new_name.json')
json_df = pd.read_json('new_name.json')

# from url
url_df = pd.read_csv("url_website")
```

## Resources:

* [pandas cheatsheet](https://github.com/pandas-dev/pandas/blob/main/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)


















