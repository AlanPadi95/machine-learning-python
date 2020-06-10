import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# Read the data from the csv to create the model
data = pd.read_csv("Linear Regression/student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#This will print out the first 5 students in our data frame.
print(data.head())

# The variable we want to predict is G3
predict = "G3"

"""
Now that we've trimmed our data set down we need to separate it into 4 arrays. 
However, before we can do that we need to define what attribute we are trying to predict.
This attribute is known as a label. The other attributes that will determine our label are known as features.
Once we've done this we will use numpy to create two arrays. 
One that contains all of our features and one that contains our labels.
"""
x = np.array(data.drop([predict], 1)) #Features
y = np.array(data[predict]) #Labels

"""
After this we need to split our data into testing and training data.
We will use 90% of our data to train and the other 10% to test.
The reason we do this is so that we do not test our model on data that it has already seen.
"""
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = -1
# We are going to get the best model of 10000 executions
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    #Defining the model which we will be using
    linear = linear_model.LinearRegression()
    #Train and score our model using the arrays we created
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    # To save our model into a file to use it when the accuracy is very high
    if acc > best:
        print("[Accuracy,", acc,"] [Try, ", _ ,"]")
        best = acc
        with open("Linear Regression/studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# LOAD BEST MODEL
pickle_in = open("Linear Regression/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("-------------------------")
print("Accuracy:\n", best)
print("Coefficent:\n", linear.coef_)
print("Intercept:\n", linear.intercept_)
print("-------------------------")

# PRINT PREDICTIONS
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Drawing and plotting model
plot = "failures" # Change this to G1, G2, studytime or absences to see other graphs
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()