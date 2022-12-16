# Scikit-Learn Notes:

## Install Necessary Modules:
```bash
pip install numpy             # for numerical analysis
pip install pandas            # for efficient dataset handlings
pip install -U scikit-learn   # for building ML models
pip install matplotlib        # for plotting the results
```

## Import Modules:
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sklearn methods are imported later as needed
```

## Load the Dataset:
Note: We are not using external *csv* files for the dataset, but we are using built-in datasets already available in the `scikit-learn` module. Since, our job here is to learn the process of building models and tuning various parameters, hyper-parameters, we can do that using these datasets.
```python
import sklearn
from sklearn import datasets
dir(datasets)                   # shows all the available datasets

# load one dataset from sklearn
iris = datasets.load_iris()     # choosing one dataset
# iris is a different type of data: sklearn.utils._bunch.Bunch
# see sklearn docs to see all things you can do with this type of data

iris.feature_names              # prints the column names of data
>>> ['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']
iris.data                       # 2D numpy array of shape (150,4)
iris.target_names               # to see what are names of target (classifications)
>>> ['setosa', 'versicolor', 'virginica']
iris.target                     # 1D numpy arra of shape (150, )
iris.DESCR                      # to know more about data within console

# loading csv dataset
import pandas as pd
df = pd.read_csv('Seed_Data.csv')
X = df.iloc[:, 0:7]                 # selecting our features set
y = df.iloc[:,7]                    # select target column
```

## Building Models using Scikit-Learn:
`SVC` stands for **Support Vector Machine**. In this we will load dataset from Kaggle website and build a simple SVC model. This dataset *seed_data* contains three variety of wheats. 210 rows, 7 features, 1 target variable (types of wheats) to be predicted. `SVC` is a supervised-classifier model. **Pipeline** is a task to combine transforming data and predicting data into one single steps for efficient work.
```python
# sklearn imports:
from sklearn import svm       # import support vector machine
from sklearn.svm import SVC   # import classfier of svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # default split: (0.75, 0.25)
from sklearn.pipeline import Pipeline                # Pipeline to use
from sklearn.preprocessing import StandardScaler     # to standarize numerical features
from sklearn.preprocessing import MinMaxScaler       # to scale data between [0-1]
from sklearn.metrics import accuracy_score           # one ways to evaluate model
from sklearn.metrics import classification_report    # to make classification report


# split dataset to train and test: randomly chosen rows in train and test here
# random_state parameter is to seed to get same values if done in future
# For csv dataset:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# For iris dataset:
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# instantiate StandardScaler with default parameters
sc = StandardScaler()

# using Pipeline to do both: preprocessing and estimating
# we have used default pipeline for logistic regression
pipe_lr = Pipeline([
    ('minmax', MinMaxScaler()),
    ('lr', LogisticRegression())
])

# fit pipeline model:
pipe_lr.fit(X_train, y_train)

# evaluate to unseen data i.e. test data set
score = pipe_lr.score(X_test, y_test)
print(score)
>>> 0.966667

# transform X_train & X_test: csv dataset: StandardScaler()
X_train = sc.fit_transform(X_train)
# for test dataset we don't need to do fit, because it already learns those parameters from train
X_test = sc.transform(X_test)

## build model & perform methods: csv dataset
clf = svm.SVC()             # using default paramters
clf.fit(X_train, y_train)
pred_clf = clf.pred(X_test)

# evaluate model: csv dataset
accuracy_score(y_test, pred_clf)
print(classification_report(y_test, pred_clf))   # prints readable table of evaluation metrics
>>> 0.9524
```

## Hyperparameter Tuning with `GridSearchCV`:
One brute force method will be to start out with the default values as we did above for few cases, and then work on to changing hyperparameters values to see how it changes overall evaluation metrics. But that will take lots of time, and also we won't be sure if we actually reach to the optimum hyperparameter combinations or not. That is where `GridSearchCV` comes to be helpful.
```python
# let's see how we can do random forest classifier algorith to tune hyperparameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV     # for grid search

# Combinations of these parameters we want to
# test our models with
param_grid = {
    'n_estimators':[2,5,10,20],
    'min_samples_split':[2,3],
    'min_samples_leaf':[1,2,3]
}

# model_including grids
grid_search = GridSearchCV(
    estimator = RandomForestClassifier(),
    param_grid = param_grid
)

# fit:
grid_search.fit(X_train, y_train)

# best params
grid_search.best_params_
>>> {'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 2}

# once best params are found, now create model and then test its predictions
clf_best_param = RandomForestClassifier(
    n_estimators=2, 
    min_samples_split=3, 
    min_samples_leaf=3
)

# fit:
clf_best_param.fit(X_train, y_train)

# predict
y_pred = clf_best_param.predict(X_test)

# accuracy
accuracy_score(y_test, y_pred)
>>> 1.0  # this case we got 100%, might be overfitting!
```

## K-Means Clustering with Scikit-Learn:
```python
from sklearn.cluster import KMeans

# prepare the data: we can think as only has features
# note: target is not present, coz we don't know that
X = np.array([
    [5, 3],
    [10, 15], 
    [15, 12],
    [24, 10],
    [30, 45], 
    [85, 70],
    [71, 80],
    [60, 78], 
    [55, 52],
    [80, 91]
])

# visualize the data. This will just show points
plt.scatter(X[:,0], X[:,1], label='True positions')

# create clusters: if we assume 2 clusters
kmeans = KMeans(n_clusters=2)

# fit
kmeans.fit(X)

# Explore the clusters: centroids & labels
print(kmeans.cluster_centers_)
print(kmeans.labels_)
>>> [[70.2 74.2]
     [16.8 17. ]]
    [1 1 1 1 1 0 0 0 0 0]

# Visualize k-means with two Clusters:
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')

# what if we choose n_clusters=3
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
>>> [[74.   79.75]
     [13.5  10.  ]
     [42.5  48.5 ]]
    [1 1 1 1 2 0 0 0 2 0]

# Visualize k-means with two Clusters:
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
# this visualization will show 3 different colors for data points

# to see the centroid along with cluster points
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black')
```

## K-Nearest Neighbors with Scikit-Learn:
* can be used to solve both **classification** and **regression** problems
* supervised algorithm, reiles on labeled input data to learn function from and predict on new unlabeled data
* assumes that nearer things share similar features, and so choose any close *K* number of neighbors' labels to decide which label to pick for the new data
    * the means of the close neighbors will be used for the regression problems
    * the mode of the close neighbors will be used for the classification problems
* KNN best use case will be in places where we need to solve problems that have the solutions that depend on similar attributes. Like in the *recommender systems*, recommending movies, articles, musics, etc.
```python
# now let's see KNN with scikit-learn
# do train_test_split, StandardScaler() to normalize data
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
# pred
y_pred = clf.predict(X_test)
# evaluation
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```