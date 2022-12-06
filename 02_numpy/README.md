# Numpy notes:

## Numpy Introduction:

Because data is growing exponentially every second of the time, data science popularity also increased, because we need many people to work on those data and find meaningful insights. Data science is a mixture of math, statistics, and computer science, problem solving, curosity, passionate to learn. Several tools exists to work on data projects, but python is rich and it is flexible, and it has large community, meaning you will be well guided to learn it.

Numpy is one of the primary packages along with others to do data science. We don't really directly interact with numpy but many packages are based on numpy as baseline. Numpy is fast and efficient to work on large datasets.

Comparing pyton list and numpy array, is not a good comparision, because, numpy array is way more efficient and faster. Because of this, almost all of the advanced packages doing ML borrows numpy data structure as their base-level.

When looking at dataset for the first time, make sure you remove the string + numbers mixed up data, for e.g. \$1200 should have removed in order to perform calculations on them. Also, numpy data-structure demands that all the data within a column of the dataset be the same-type.

## import numpy with common alias
```python
import numpy as np
```

## Initializing Numpy Arrays
```python
## From list to array
a = [1,2,3,4]
numpy_a = np.array(a, dtype='int16')     # dtype of int16 or int8 reduces memory size further

# 2d-array - hard coding
b = np.array([
    [1,2,3,4],
    [5,6,7,8]
])
print(b)

>>> [[1 2 3 4]
     [5 6 7 8]]

## To create fast arrays of any sizes
np.zeros((5,6))                     # returns matrix of zeros of size (5 rows, 6 cols)
np.ones((3,3))                      # returns matrix of ones of size (3,3)
np.full((5,3), 8)                   # matrix filled with any number
np.eye(5)                           # identity matrix of size (5,5)
np.random.rand(4,3)                 # random matrix of values betn 0-1 size (4,3)
np.random.rand(4,3)*10              # scale up, matrix values betn 0-10 size (4,3)

# by copying
a = np.array([1,2,3])
b = np.copy()           # copy otherwise changing b will change a, unknowingly
```

## Methods and Attributes on arrays
```python
# Get Dimension
b.ndim
>>> 2    # for 2 dimensional

# Get Shape
b.shape
>>> (2,3)

# Get Type
print(b.dtype)
>>> int64

# Get Size
print(b.nbytes)
>>> 64
```

## Numpy Indexing and Accessing Elements
```python
# first see how we access elements in list
python_list = [
    ['apple', 'orange', 'banana'], 
    ['panda', 2, 100]]
python_list[1][2]
>>> 100

# numpy array
a = np.array([
    [1,2,3,4,5,6,7],
    [8,9,10,11,12,13,14]
])

# get specific element [row, col]
a[:2, :-2]              # first two row, all cols except last 2

# get specific row
a[0,:]                  # row 0, all cols

# get specific col
a[:, 4]                 # all rows, col 4(index)

# modify array
a[:,2] = [1,2]          # first access it, and put the values

# Example of indexing into 3D array
c = np.array([
    [[1,2],[3,4]],
    [[5,6],[7,8]]
  ])
# access 1st row, 1st col, 0th value
print(c[1, 1, 0])
```

## Numpy Shape-shifting
```python
# concatenating arrays
arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
np.concatenate((arr1, arr2))        # axis=None, by default flattens
>>> array([1, 2, 3, 4, 5, 6, 7, 8])

# vertical stacking
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
np.vstack([v1,v2])       
>>> array([[1, 2, 3, 4],
           [5, 6, 7, 8]])

# horizontal stack
h1 = np.ones((4,5))
h2 = np.full((4,2), 2)
np.hstack((h1,h2))        # or, np.concatenate((h1,h2), axis=1)
>>> array([[1., 1., 1., 1., 1., 2., 2.],
           [1., 1., 1., 1., 1., 2., 2.],
           [1., 1., 1., 1., 1., 2., 2.],
           [1., 1., 1., 1., 1., 2., 2.]])

# Reshape array
a = np.full((4,8),3)
a_flat = a.flatten()           # all values are flatten to 1D
b = np.reshape(a_flat, (2,16)) # 4x8=32=2x16 total nb. should match to reshape
```

## Numpy Math & Broadcasting
```python
a = np.array([1,2,3,4])
print(a + 7)                # simply adds 7 to all values
print(a * 3)                # multiply 3 to all values
print(a / 2)                # divides by 2 to all values
b = np.array([4,5,6,7])
print(a + b)                # to add element-by-element, same shape!
print(a * b)                # multiply element-by-element, same shape!
>>> [4 10 18 28]
np.sin(a)                   # taking sin, cos, tan, etc

# matrix-multiplication
a = np.array([
    [1,2],
    [2,3]
])
>>> (2,3)

b = np.array([
    [1,2],
    [2,3],
    [3,3]
])
>>> (3,2)
# to be compatible to have matrix-multiplication
# inner-dim should match: (2,2)x(2,3) => (2,3)
# so, we need to transpose b to do this
np.matmul(a,b.transpose())
>>> [[5 8 9]
     [8 13 15]]
```

## Load data from file into Numpy
```python
# assuming csv file with header row, and delimiter is comma
filedata = np.genfromtxt('path/to/file.csv', delimiter=',', skip_header=1)
filedata.shape          # returns the (r,c)  shape of file
```

## Load dataset with numpy
Numpy as allows to load csv files (better is by using pandas!)

```python
from numpy import genfromtxt

# First row is: id, airbnb_id, price, latitude, longitude
matrix_1 = genfromtxt(
    csv_unzipped_file_1, delimiter=";", skip_header=False, dtype="unicode"
)
matrix_2 = genfromtxt(
    csv_unzipped_file_2, delimiter=";", skip_header=False, dtype="unicode"
)
```
## change dimensions of matrix array

Sometimes we need to change the dimensions of the matrix

```python
np.ravel() # returns a view of numpy matrix collapsed into one dim
np.flatten() # returns a copy of numpy matrix collapsed into one dim
np.split() # provides sub-arrays, and tell which axis to split on
np.resize() # change the size of array
T or np.transpose() # inverst the axis of numpy matrix
np.flip()  # reverse the order of elements along the given axis
```

## merge two csv files

Data files are generally splitted into many small files (csv) we should merge them. Let's say we have two files named
```
Matrix 1 has dimensions:  (4097, 5) 
Matrix 2 has dimensions:  (2078, 5)
```
To merge, we use `np.concatenate` so total shape will be `(6175,5)`.

To merge:
```python
matrix = np.concatenate((matrix_1, matrix_2), axis=0)
print("Matrix has dimensions: ", matrix.shape)

# gives the desired shape now
```
`axis=0` ensures that the merging happens row-wise. i.e. we can think of stacking two datasets on top of each other. If we had `axis=1`, doesn't make sense here though, merging would happen side-by-side. also `concatenate` joins the matrix on the existing axis.

`np.stack()` joins numpy arrays by creating a new axis.

To select and view few entries:
```python
print(matrix[:5, :]) # row from 0:5, and all columns :
```

## slice and view matrix array

Other ways to view data from numpy array slicing:
```python
# Selection & slicing operations: [row], [row, columns], [row][column], [axis_0, axis_1, axis_2, ...], etc...
print("One entry\t\t\t:", matrix[6000])
print("One entry, a few elements\t:", matrix[6000, 1:3])
print("One element\t\t\t:", matrix[5000, 2])
print("One element\t\t\t:", matrix[5000][2])
```

## modify the matrix array
To drop first column:
```python
# Drop the first column
matrix = matrix[:, 1:]

# First five rows of updated matrix
print(matrix[:5, :])
```

Unicode data type means that it is a string. We need to change to float to be able to do math. But even before that, we need to get rid of pesky strings mixed up with numerical values as in \$1200. Since, we stacked two matrix, both of them has the header row, we need to get rid of. they are in loc: `matrix[0]` and `matrix[4097]`. So, how to delete these two rows?
```python
# Start with the highest index to prevent incorrect deletion
matrix = np.delete(matrix, (4097), axis=0)
matrix = np.delete(matrix, (0), axis=0)
```
Otherwise you will delete the wrong entry (i.e., once the top row is deleted, row 4097 becomes row 4096, so a different row 4097 will be deleted).

other related functions:
```python
np.insert() # insert a value into an existing numpy matrix
np.append() # add a value or row at the end of an existing numpy matrix
```
## remove commas, strings, dollar signs

Next let's work on removing commas, periods, and dollar signs attached into the numerical values.

We can do this:
```python
matrix[np.char.find(matrix, ",") > -1]

# this will give
array(['$1,036.00', '$1,160.00', '$1,290.00', '$2,000.00', '$1,200.00',
       '$1,975.00', '$1,600.00', '$1,200.00', '$1,000.00', '$1,000.00',
       '$1,000.00', '$1,000.00', '$1,000.00', '$1,000.00', '$2,350.00',
       '$1,186.00', '$1,194.00', '$2,500.00', '$1,000.00', '$1,000.00',
       '$1,000.00', '$1,000.00', '$1,000.00'], dtype='<U18')
```
other related methods:

```python
np.char.endswith() # find elements ending with specified characters
np.char.startswith() # find elements starting with specified characters
```

Can we remove command and dollar sign?
```python
matrix = np.char.replace(matrix, "$", "")
matrix = np.char.replace(matrix, ",", "")
```

Check if the above command removed what we wanted to remove:
```python
matrix[(np.char.find(matrix, "$") > -1) | (np.char.find(matrix, ",") > -1)]
```
No output, meaning it worked!

Finally, we can now convert to float from unicode using `astype(float)`
```python
matrix = matrix.astype(float)
```

Tips: if data appeared in scientific notation and it's hard to read, do this for easier to read
```python
# Disable scientific notation
np.set_printoptions(suppress=True)
```

## what is broadcasting in numpy array?

Broadcasting is a rich topic, and also very important concept. This helps to do the large numerical operations efficiently. If there were no vectorizing (and broadcasting) then, we would have to depend on the `for` loops to do the same operation, and that would be a huge complexity in terms of computational power and also time. see image below:

<p align="center">
  <img src="https://github.com/sbdrchauhan/python-DS-ML/blob/master/07_MOOC/CoRise_two_weeks_py_for_ds/Images/broadcasting.png" width=50% height=50%>
</p>

Broadcasting is a computationally and memoery-efficient way of calculating multiple equations at once, which is what we need for most machine learning algorithms.

Let's say we want to multiply second column of our matrix that contains the price, but now we also want to add the inflation of ~10\% to that price column only.
```python
# Add 10% to the prices by:
# - selecting a subset of the matrix
# - using broadcasting to multiply
matrix[:, 1] = matrix[:, 1] * 1.10
```
Some more related methods:
```python
* or np.multiply()  # multiply elements or matrices
/ or np.divide()  # divide elements or matrices
+ or np.add()  # sum elements or matrices
- or np.subtract()  # subtract elements or matrices
** or np.power()  # raise the first elemnent or matrix to the power
    # of the second element
% or np.remainder()  # get the remainder of the first-ordered element
  # or matrix, using second-ordered element as the divisor
// or np.floor_divide()  # divide two elements or matrices, with the
  # result rounded down to the nearest integer
    
np.sqrt() # calculate the square root of each element in the matrix
np.cos() # cosine of each element
np.sin()
np.tan()
np.absolute()
np.log()  # calcualte natural logarithm for each element in the matrix
np.exp()
```

## Conditional selection
`matrix[matrix[:,1]<20]` we select second column and see only those values that are less than 20, and use it as mask to select all other columns of the total matrix that satisfies the mask conditions.
```python
# Select rows for which the price_usd column is lower than 20
matrix[matrix[:, 1] < 20]
```
Some related methods:
```python
np.amin() # returns the min element of the matrix
np.amax()  # returns the max element of the matrix
np.sum()  # returns the sum of the matrix
np.average() # returns the mean/average of the matrix
np.median()  # return the median of the matrix
np.mean()  # return mean/avg of the matrix
np.var()  # return the variance of the matrix
np.std()  # return the standard deviation of the matrix
```

## Currency Converter
We need to add two more columns to our matrix; one that shows price in Indian currency and one for Chinese currency (we have USD given in data).
```python
from currency_converter import CurrencyConverter

cc = CurrencyConverter()
indian_rupee = cc.convert(100, "USD", "INR")
chinese_yuan = cc.convert(100, "USD", "CNY")

print("We have 100 dollars. When we convert it to:\n")
print("Indian rupees, we get:\t", indian_rupee)
print("Chinese yuan, we get:\t ", chinese_yuan)

# it shows:
We have 100 dollars. When we convert it to:

Indian rupees, we get:	 7984.660737483762
Chinese yuan, we get:	  686.2296392525233
```
Let's perform the conversion in a vectorize way. First define the function that does the conversion:
```python
def convert_usd_to_inr_or_cny(dollar: float, currency: str):
    if currency == "INR":
        return cc.convert(dollar, "USD", "INR")
    else:
        return cc.convert(dollar, "USD", "CNY")
```
Then, we can vectorize the function, then apply it to the matrix.
```python
%%timeit -r 4 -n 100

# (Semi-)vectorize a Python function
convert_vec = np.vectorize(convert_usd_to_inr_or_cny)

# Apply the function, use timeit
convert_vec(matrix[:, 1], "INR"), convert_vec(matrix[:, 1], "CNY")
```
This method of vectorizing and applying is much efficient and faster, had we have done it using typical `for` loop. But, it still lacks the true vectorize that numpy is capable of performing.
```python
%%timeit -r 4 -n 100

inr_rate = cc.convert(1, "USD", "INR")
cny_rate = cc.convert(1, "USD", "CNY")

matrix[:, 1] * inr_rate, matrix[:, 1] * cny_rate
```
In this small dataset, it didn't matter much, but always vectorize when dealing with the massive datasets or slow functions. **vectorization is worth it!**