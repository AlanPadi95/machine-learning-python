# Overview
This series is designed to teach you the fundamentals of 
machine learning with python. It will start by introducing some basic machine learning algorithms 
and slowly move into more advanced topics like neural networks.

# What You’ll Learn
This series is packed full of valuable information. 
You will learn and understand the following after this [tutorial](https://techwithtim.net/tutorials/machine-learning-python/):

* [Linear Regression](#linear-regression)
* [K-Nearest Neighbors](#k-nearest-neighbors)
* [Support Vector Machines](#support-vector-machines)
* K-Means
* Neural Networks
* Conventional Neural Networks

# Pre-requisites
This is NOT a beginner tutorial and I will not be teaching python syntax.

* Intermediate/Advanced Programming Knowledge
* Experience With Python 3 Syntax

# Requirements

To make this tutorial more easy, there are a few ways to setup the environment depending of your Operative System:
* [Windows](https://gist.github.com/marcelotm23/461540ed5b7a19277dc24432e2ef3d3c/)
* [MacOSX](https://gist.github.com/AlanPadi95/73f1baa187047d9ba7fc22ef0f2f7537)

# Conclusions

## Linear Regression
Linear Regression is an algorithm that every Machine Learning enthusiast 
must know and it is also the right place to start for people
who want to learn Machine Learning as well.
  
It is one of the most common machine learning processes in the world and it helps prepare businesses in a volatile and dynamic environment.
 
### Advantages and Disadvantages of Linear Regression

#### Advantages
Linear regression is an extremely simple method. It is very easy and intuitive to use and understand.
A person with only the knowledge of high school mathematics can understand and use it. 
In addition, it works in most of the cases. Even when it doesn’t fit the data exactly, 
we can use it to find the nature of the relationship between the two variables.

#### Disadvantages
* By its definition, linear regression only models relationships between dependent and independent variables that are linear.
 It assumes there is a straight-line relationship between them which is incorrect sometimes. 
 Linear regression is very sensitive to the anomalies in the data (or outliers).
* Take for example most of your data lies in the range 0-10.
 If due to any reason only one of the data item comes out of the range, say for example 15,
  this significantly influences the regression coefficients.
* Another disadvantage is that if we have a number of parameters than the number of samples available 
then the model starts to model the noise rather than the relationship between the variables.

### Applications of Linear Regression
Linear regression is a powerful statistical technique that can generate insights on consumer behavior, 
help to understand business better, and comprehend factors influencing profitability. 
It can also be put to service evaluating trends and forecasting data in a variety of fields. 
We can use linear regression to solve a few of our day-to-day problems related to supporting decision making, 
minimizing errors, increasing operational efficiency, discovering new insights, and creating predictive analytics.


 ## K-Nearest Neighbors
The k-nearest neighbors (KNN) algorithm is a simple, supervised machine learning algorithm 
that can be used to solve both classification and regression problems. It’s easy to implement
and understand, but has a major drawback of becoming significantly slows as the size of that
data in use grows.

KNN works by finding the distances between a query and all the examples in the data, selecting 
the specified number examples (K) closest to the query, then votes for the most frequent label 
(in the case of classification) or averages the labels (in the case of regression).

In the case of classification and regression, we saw that choosing the right K for our data is
done by trying several Ks and picking the one that works best.

### Advantages and Disadvantages of KNN

#### Advantages
* It is very simple algorithm to understand and interpret.
* It is very useful for nonlinear data because there is no assumption about data in this algorithm.
* It is a versatile algorithm as we can use it for classification as well as regression.
* It has relatively high accuracy but there are much better supervised learning models than KNN.

#### Disadvantages
* It is computationally a bit expensive algorithm because it stores all the training data.
* High memory storage required as compared to other supervised learning algorithms.
* Prediction is slow in case of big N.
* It is very sensitive to the scale of data as well as irrelevant features.

### Applications of KNN
The following are some of the areas in which KNN can be applied successfully:

* **Banking System**: KNN can be used in banking system to predict weather an individual is fit for loan approval? 
Does that individual have the characteristics similar to the defaulters one?

* **Calculating Credit Ratings**: KNN algorithms can be used to find an individual’s credit rating 
by comparing with the persons having similar traits.

* **Politics**: With the help of KNN algorithms, we can classify a potential voter into various 
classes like “Will Vote”, “Will not Vote”, “Will Vote to Party ‘Congress’, “Will Vote to Party ‘BJP’.

Other areas in which KNN algorithm can be used are **Speech Recognition**, **Handwriting Detection**, **Image Recognition** and **Video Recognition**.


## Support Vector Machines
A SVM has a large list of applicable uses.
However, in machine learning it is typically used for classification.
It is a powerful tool that is a good choice for classifying complicated data
 with a high degree of dimensions(features). 
Note that K-Nearest Neighbors does not perform well on high-dimensional data.

### Advantages and Disadvantages of SVM

#### Advantages
* SVM works relatively well when there is clear margin of separation between classes.
* SVM is more effective in high dimensional spaces.
* SVM is effective in cases where number of dimensions is greater than the number of samples.
* SVM is relatively memory efficient

#### Disadvantages
* SVM algorithm is not suitable for large data sets.
* SVM does not perform very well, when the data set has more noise i.e. target classes are overlapping.
* In cases where number of features for each data point exceeds the number of training data sample , the SVM will under perform.
* As the support vector classifier works by putting data points, above and below the classifying hyper plane there is no probabilistic explanation for the classification.

### Applications of SVM
The aim of using SVM is to correctly classify unseen data. SVMs have a number of applications in several fields.
Some common applications of SVM are:

* **Face detection**: SVMc classify parts of the image as a face and non-face and create a square boundary around the face.
* **Text and hypertext categorization**: SVMs allow Text and hypertext categorization for both inductive and transductive models. They use training data to classify documents into different categories. It categorizes on the basis of the score generated and then compares with the threshold value.
* **Classification of images**: Use of SVMs provides better search accuracy for image classification. It provides better accuracy in comparison to the traditional query-based searching techniques.
* **Bioinformatics**: It includes protein classification and cancer classification. We use SVM for identifying the classification of genes, patients on the basis of genes and other biological problems.
* **Protein fold and remote homology detection**: Apply SVM algorithms for protein remote homology detection.
* **Handwriting recognition**: We use SVMs to recognize handwritten characters used widely.
* **Generalized predictive control(GPC)**: Use SVM based GPC to control chaotic dynamics with useful parameters.