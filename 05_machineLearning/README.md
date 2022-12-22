# :books: Notes on ML practices :books:

Machine learning gives computers ability to learn from data without being explicity programmed to do so. This is done by training the code on large amount of data and then from this learning make some predictions on new data. Since our computers are now fast enough and powerful to deal with big data, we can now achieve this and so the field has increased interest lately. Because now computers are fast and algorithms are improved so it predicts faster and do lots of work previously could not achieve.

# Supervised Learning
It is a type of ML algorithm where labeled training data is used to train or learn a function that maps the input data to the desired output. Example include if we have image classification task, we have 1000s of rows of data where each row is one image feature and also has the label to tell algorithm what picture it is. Then, this is supervised learning in that we have label output already to learn from and apply the learning to the newer dataset. Supervised learning task can be used for **classification, regression, and predictions**. Some of the common algorithm used in Supervised learning are:
* decision trees,
* support vector machines,
* neural nets., etc

## Linear Regression
It is widely used statistical method for modeling the relationship between a dependent variable and one or more independent variable. Here we try to find the line of best fit that accurately describes the relationship between the dependent and independent variables. The line of best fit is that line that minimizes the error between the predicted and the actual values. Example of use, predicting house prices given all the features of house, stock prices based on historical data, etc. Linear regression assumes that the relationship between the dependent and independent variables are linear. Any non-linear relationships can't be solved by this method, so in that case, we need to use other non-linear methods.

## Decision Trees
It is a type of supervised learning algorithm that can be used for classification and regression tasks. It works by dividing the data into smaller and smaller groups, based on the value of certain features, until each group is "pure", meaning all the data in the group belongs to the same class. The algorithm sets the rules to determine which features to use for dividing the data, aiming to maximize the "information gain" at each step. Once the decision tree has been trained on a dataset, it can be used to make predictions on new data by feeding it through the tree and following the branches based on the values of the features. Decision trees have several advantages, including the ability to handle both numerical and categorical data and the ability to interpret and visualize the rules used to make predictions. However, they can also be prone to overfitting and may not perform as well on complex or non-linear datasets.

## K-Nearest Neighbor
KNN is a popular and simple machine learning algorithm for classification and regression tasks. In KNN, the model makes predictions or decisions based on the similarity of new data to the training data. To make a prediction, the model calculates the distance between the new data and the training data, and selects the K training points that are closest to the new data. The model then uses the labels or outputs of the K nearest points to make the prediction or decision for the new data. KNN can be used for a variety of tasks, such as predicting the type of a flower based on its characteristics, or identifying the author of a piece of text based on their writing style. KNN can be sensitive to the choice of K and the distance metric, and it can perform poorly for high-dimensional or noisy data.


# Unsupervised Learning
Unsupervised learning is a type of machine learning algorithm that uses unlabeled data to learn patterns and relationships in the data. The goal is to find structure and meaning in the data, without the guidance of labeled training data. Unsupervised learning algorithms can be used for a variety of tasks, such as clustering data into groups, identifying anomalies or outliers in the data, or reducing the dimensionality of the data. Some common unsupervised learning algorithms include k-means clustering, principal component analysis, and autoencoders. Unsupervised learning can provide valuable insights into the structure and relationships in the data, and can be used in combination with supervised learning to improve the performance of machine learning models. However, unsupervised learning can be challenging to apply and interpret, and it can be sensitive to the choice of algorithm and parameters.

## K-Means Clustering
K-means clustering is a popular and simple unsupervised learning algorithm for clustering data into groups. In K-means clustering, the goal is to partition the data into K clusters, where each cluster is defined by its center or centroid, and each data point belongs to the cluster with the closest centroid. To find the clusters, the K-means algorithm iteratively updates the centroids and assigns the data points to the closest centroids, until the centroids converge and the assignments of the data points do not change. To find the clusters, the K-means algorithm iteratively updates the centroids and assigns the data points to the closest centroids, until the centroids converge and the assignments of the data points do not change. K-means clustering is simple to implement and fast to run, and it can provide good results for many types of data. However, K-means clustering is sensitive to the choice of K and the initialization of the centroids, and it can perform poorly for non-linearly separable or non-uniformly distributed data.

# Reinforcement Learning
Reinforcement learning is a third type of ML, is a type of machine learning algorithm that uses a trial-and-error approach to learn how to maximize a reward or goal. In reinforcement learning, the model is an agent that interacts with an environment and takes actions to achieve a reward or goal. The agent receives feedback in the form of rewards or punishments, which it uses to update its decision-making strategy and improve its performance over time. Reinforcement learning can be used for a variety of tasks, such as playing games, controlling robots, or optimizing resource allocation. Some popular reinforcement learning algorithms include Q-learning, Monte Carlo methods, and deep reinforcement learning. Reinforcement learning is a powerful and flexible technique that can learn complex behaviors and adapt to changing environments. However, reinforcement learning can be difficult to apply and optimize, and it can require a lot of data and computational resources to achieve good performance. Overall, reinforcement learning is an exciting and promising area of machine learning that has the potential to solve many complex and challenging problems.

## Ensemble learning
Ensemble learning is a machine learning technique that combines multiple models to improve the performance and robustness of the final model.In ensemble learning, the individual models, called base models or weak learners, are trained on the same data, but the data input varies from model to model, and their predictions are combined using a combination rule, such as majority voting, averaging, or weighted averaging. Ensemble learning can be used for a variety of tasks, such as classification, regression, and clustering. Some common ensemble learning algorithms include bagging, boosting, and stacking. Ensemble learning can provide better performance and more robust predictions than individual models, and it can be used to reduce overfitting and improve the generalizability of the final model. However, ensemble learning can be computationally expensive, and it can require careful design and tuning of the base models and the combination rule.



<p align="center">
    <img src="./images/ensemble.png" />
</p>






## Things to consider when choosing ML models to solve a problem
* Type of problem:
    - Is is classification, regression, or clustering? Based on the answer you have different models you should use.
* Size and Quality of data:
    - Because different model capability differ based on both the size and the quality of data, given the condition of data in your hand, you can further narrow down you choice of ML models to use.
* Computational Resources available:
    - Based on how much computational power you have, and how fast you need preliminary answer to the problem, you need to decide on model, again, there are many algorithms that need lot more computational power than others do.
* Accuracy and Performance Metric:
    - Different model accuracy and performance varies, so depending on accurate or precise you want your answer, you can choose from simple to complex models to train your data.

All in all, there is no one-size-fit-all model that will work for all the problems in the world, you need to decide on the choice of model to used based off your requirements and available resources at hand.

## Tips to become successful ML engineer
* Develop strong foundation in ML concepts and techniques, including learning the different models syntax and its use, data processing, and model evaluations.
* Gain experience with a variety of tools and frameworks, including popular open-source libraries such as TensorFlow, and PyTorch.
* Build a portfolio projects that showcases your skills and capabilities, and consider earning relevant certifications to demonstrate your expertise.
* Network with other professionals in the field, attend conferences and workshops, and stay up-to-date on the latest development and trends in the field.
* Be wlling to continuously learn and adapt, as the field of ML is constantly evolving and new technologies and appproaches are constantly emerging.

## References:
[Dan twitter](https://twitter.com/DanKornas)