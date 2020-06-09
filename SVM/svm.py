# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics

"""However, now that we have learned this we will use the data sets that come with sklearn.
We will be using a breast cancer data set. It consists of many features describing a tumor
 and classifies them as either cancerous or non cancerous."""
cancer = datasets.load_breast_cancer()

print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)

x = cancer.data  # All of the features
y = cancer.target  # All of the labels

# It is time to split it into training and testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# If we want to have a look at our data we can print the first few instances.
print(x_train[:5], y_train[:5])

""" Change the Kernel to another option 
- linear
- poly
- rbf
- sigmoid
- precomputed
"""
clf = svm.SVC(kernel="linear")

"""
By default our kernel has a soft margin of value 1.
This parameter is known as C.
We can increase C to give more of a soft margin, 
we can also decrease it to 0 to make a hard margin.
Playing with this value should alter your results slightly.
"""
clf_with_margin = svm.SVC(kernel="linear", C=2)

clf.fit(x_train, y_train)
clf_with_margin.fit(x_train, y_train)

y_pred = clf.predict(x_test)
y_pred_margin = clf_with_margin.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
acc_margin = metrics.accuracy_score(y_test, y_pred_margin)

print(acc)
print(acc_margin)