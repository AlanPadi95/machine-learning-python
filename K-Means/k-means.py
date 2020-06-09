import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

"""
We are going to load the data set from the sklearn module
and use the scale function to scale our data down.
We want to convert the large values that are contained 
as features into a range between -1 and 1 to simplify calculations
and make training easier and more accurate.
"""
digits = load_digits()
data = scale(digits.data)
y = digits.target


"""
We also define the amount of clusters by creating a variable k
and we define how many samples and features we have by getting the data set shape.
"""
k = 10
samples, features = data.shape

"""
To score our model we are going to use a function from the sklearn website.
It computes many different scores for different parts of our model. 
"""
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

"""
Finally to train the model we will create a K Means classifier,
 then pass that classifier to the function we created above to score and train it.
"""
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)