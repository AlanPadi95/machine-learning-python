import sklearn
from sklearn import svm
from sklearn import datasets

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
