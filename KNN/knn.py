"""
Introduction to KNN
KNN stands for K-Nearest Neighbors. KNN is a machine learning algorithm used for classifying data.
Rather than coming up with a numerical prediction such as a students grade or stock price it attempts to classify data into certain categories.
We will be using this algorithm to classify cars in 4 categories based upon certain features.
"""

import  sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Loading Data
data = pd.read_csv("car.data")
print(data.head())

"""
As you may have noticed much of our data is not numeric. 
In order to train the K-Nearest Neighbor Classifier we must convert any string data into some kind of a number. 
Luckily for us sklearn has a method that can do this for us.

We will start by creating a label encoder object and then use that to encode each column of our data into integers.
"""
le = preprocessing.LabelEncoder()

"""The method fit_transform() takes a list (each of our columns) and will return to us an array containing our new values"""
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
print(buying)

# VARIABLE TO PREDICT
predict = "class"

"""
We can do that we need to define what attribute we are trying to predict.
This attribute is known as a label. The other attributes that will determine our label are known as features.
Once we've done this we will use numpy to create two arrays. 
One that contains all of our features and one that contains our labels.
"""

x = list(zip(buying, maint, door, persons, lug_boot, safety)) # features
y = list(cls) # labels

# We will split our data into training and testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# print(x_train, y_test)

"""
Creating a KNN Classifier is almost identical to how we created the linear regression model.
The only difference is we can specify how many neighbors to look for as the argument n_neighbors.
"""
neighbors = 9
model = KNeighborsClassifier(n_neighbors=neighbors)
# Train the model
model.fit(x_train, y_train)
# Get the accuracy
acc = model.score(x_test, y_test)

"""
Now, we are going to test the model
The KNN model has a unique method that allows for us to see the neighbors of a given data point.
We can use this information to plot our data and get a better idea of where our model may lack accuracy. 
"""
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

    # The .kneighbors method takes 2D as input,
    # this means if we want to pass one data point
    # we need surround it with [] so that it is in the right shape.
    """
    Parameters: The parameters for .neighbors are as follows: data(2D array), # of neighbors(int), distance(True or False)
    Return: This will return to us an array with the index in our data of each neighbor.
     If distance=True then it will also return the distance to each neighbor from our data point.
    """
    n = model.kneighbors([x_test[x]], neighbors, True)

    print("N: ", n)
    print()
    print()

print("Accuracy: ", acc)
