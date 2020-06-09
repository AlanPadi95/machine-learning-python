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
* [K-Means Clustering](#k-means-clustering)
* Neural Networks
* Conventional Neural Networks

# Pre-requisites
This is NOT a beginner tutorial and I will not be teaching python syntax.

* Intermediate/Advanced Programming Knowledge
* Experience With Python 3 Syntax

# Requirements

To make this tutorial more easy, there are a few ways to setup the environment depending of your Operative System:
* [Windows](https://gist.github.com/marcelotm23/461540ed5b7a19277dc24432e2ef3d3c/) + PowerShell
* [MacOSX](https://gist.github.com/AlanPadi95/73f1baa187047d9ba7fc22ef0f2f7537) + Ansible

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

## K-Means clustering
K Means clustering is an unsupervised learning algorithm that attempts to divide our training data
into k unique clusters to classify information.
This means this algorithm does not require labels for given test data. 
It is responsible for learning the differences between our data points
and determine what features determining what class.

### How K-Means Clustering Works
The K-Means clustering algorithm is a classification algorithm that follows the steps outlined below
to cluster data points together. It attempts to separate each area of our high dimensional space
into sections that represent each class. When we are using it to predict it will simply find 
what section our point is in and assign it to that class.

**Step 1**: Randomly pick K points to place K centroids.

**Step 2**: Assign all of the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to.

**Step 3**: Average all of the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position.

**Step 4**: Reassign every point once again to the closest centroid.

**Step 5**: Repeat steps 3-4 until no point changes which centroid it belongs to.

### Advantages and Disadvantages of K-Means Clustering

#### Advantages
* Relatively simple to implement.
* Scales to large data sets.
* Guarantees convergence.
* Can warm-start the positions of centroids.
* Easily adapts to new examples.
* Generalizes to clusters of different shapes and sizes, such as elliptical clusters.

#### Disadvantages

* Choosing manually.
> Use the “Loss vs. Clusters” plot to find the optimal (k), as discussed in Interpret Results.

* Being dependent on initial values.
> For a low , you can mitigate this dependence by running k-means several times with different initial values
> and picking the best result.
> As  increases, you need advanced versions of k-means to pick better values of the initial centroids (called k-means seeding).
> For a full discussion of k- means seeding see, 
> A Comparative Study of Efficient Initialization Methods for the K-Means Clustering Algorithm 
> by M. Emre Celebi, Hassan A. Kingravi, Patricio A. Vela.


* Clustering data of varying sizes and density.
> k-means has trouble clustering data where clusters are of varying sizes and density. 
> To cluster such data, you need to generalize k-means as described in the Advantages section.

* Clustering outliers.
> Centroids can be dragged by outliers, or outliers might get their own cluster instead of being ignored.
> Consider removing or clipping outliers before clustering.

* Scaling with number of dimensions.
> As the number of dimensions increases, a distance-based similarity measure converges to a constant value between any given examples.
> Reduce dimensionality either by using PCA on the feature data, or by using “spectral clustering” 
> to modify the clustering algorithm as explained below.

## Applications of K-Means Clustering
K-Means algorithm is very popular and used in a variety of applications such as **market segmentation**,
**document clustering**, **image segmentation**, **image compression**, etc.

The goal usually when we undergo a cluster analysis is either:

* Get a meaningful intuition of the structure of the data we’re dealing with.

* Cluster-then-predict where different models will be built for different subgroups
 if we believe there is a wide variation in the behaviors of different subgroups.
 An example of that is clustering patients into different subgroups and build a model
 for each subgroup to predict the probability of the risk of having heart attack.