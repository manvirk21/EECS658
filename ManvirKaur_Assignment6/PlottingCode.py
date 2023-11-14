'''
File Name: imbalanced.py
Description: Uses unsupervised machine learning to anaylize the iris dataset
Author: Manvir Kaur
KUID: 3064194
Date: 11-09-23
Inputs: iris.csv dataset
Outputs: accurarcy scores and confusion matrixes for 4 different unsupervised methods
Sources: Dr. Johnson's lecture notes, debugging with ChatGPT
'''
 
# Load the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

import warnings
warnings.filterwarnings("ignore")


# Load the dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Create arrays for features and classes
array = dataset.values
x = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names

# Using  `LabelEncoder` class from  `preprocessing` module of scikit-learn to encode target variable `y`.
array = preprocessing.LabelEncoder()
array.fit(y)
y = array.transform(y)

#PART 1

#Find the optimum k value:
wcss = []

''' Performing a loop from 1 to 20 and for each iteration, it is creating a KMeans clustering model
with a different number of clusters. The `kmeans.inertia_` attribute is used to calculate the within-cluster 
sum of squares (WCSS) for each model, which is a measure of how internally coherent the clusters are. 
The WCSS values are then appended to the `wcss` list. This process helps in finding the optimal number 
of clusters by identifying the "elbow" point in the plot of WCSS versus the number of clusters. '''
for i in range(1,21):
    # Creating an instance of the KMeans clustering algorithm.
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 100, n_init = 10, random_state = 0)
    # Fitting the KMeans clustering algorithm to the data `x`. It means that the algorithm is learning the 
    # patterns and structure in the data in order to create clusters.
    kmeans.fit(x)
    ''' Calculating and storing the within-cluster sum of squares (WCSS) for each value of k in the range. 
    By calculating the WCSS for different values of k, we can determine the optimal number of clusters 
    for the KMeans algorithm. '''
    wcss.append(kmeans.inertia_)
    
# Creating a line plot of the WCSS values against the number of clusters (k) ranging from 1 to 20.
# This plot helps in visualizing the relationship between the number of clusters and the WCSS, which 
# can be used to determine the optimal number of clusters for the KMeans algorithm.
plt.plot(range(1,21), wcss)
plt.title('Reconstruction Error vs. k') # Setting the title of the plot
plt.xlabel('k') # Setting the label for the x-axis
plt.ylabel('Reconstruction Error') # Setting the label for the y-axis
# Displays the currently active figure. It is used to show the plot that has been created
plt.show()

name = "Kmeans"
# use k-means clustering to cluster data into 3 clusters
''' This value is used as the number of clusters for the algorithms. It is determined by finding the 
"elbow" point in the plot of WCSS. The elbow point is the point of inflection in the plot, indicating 
the optimal number of clusters. In this case, the algorithms will create 3 clusters. '''
elbow_k = 3
''' Creating an instance of the KMeans clustering algorithm with the specified number of clusters. The
algorithm is then fitted to the data `x`, which means it learns the patterns and structure in the
data to create the clusters. The `init='k-means++'` parameter specifies the initialization method
for the algorithm, and `random_state=42` ensures reproducibility of the results. The resulting
`final_kmeans` object can be used to make predictions or analyze the clusters.'''
final_kmeans = KMeans(n_clusters=elbow_k, init='k-means++', random_state=42).fit(x)
# use the predict() method and clusters for k = elbow_k to classify the entire iris data set.
# x = usual iris data set from iris.csv
prediction = final_kmeans.predict(x)
# Match the kmean labels and the truth labels such that the number of true-positive predictions is maximized
k_labels_matched = np.empty_like(prediction)
''' Iterating over the unique values in the `prediction` array. Loop through each unique cluster label assigned 
by the clustering algorithm. Allows further analysis or processing to be performed on each cluster separately.'''
for i in np.unique(prediction):
    # ...find and assign the best-matching truth label
    match_nums = [np.sum((prediction == i) * (y == t)) for t in np.unique(y)]
    k_labels_matched[prediction == i] = np.unique(y)[np.argmax(match_nums)]
#print confusion matrix 
print('Confusion matrix of', name, ': ')
cmatrix = np.array(confusion_matrix(y, k_labels_matched))
print(cmatrix, end='\n\n')
#print accuracy
print('Overall accuracy of', name, ': ', end=' ')
accuracy_score_ = np.trace(cmatrix) / np.sum(cmatrix)
print(accuracy_score_, end='\n\n');



# Part 2
#AIC
#Find the optimum k value:
wcss = []

''' Performing a loop from 1 to 20. For each iteration, create a Gaussian Mixture Model (GMM) with a 
different number of components (clusters). The AIC (Akaike Information Criterion) value is then 
calculated for each GMM model using the `gmm.aic(x)` method. The AIC is a measure of the model's quality 
and complexity, with lower values indicating better models. The AIC values are then appended to the `wcss` 
list, which is used to store the AIC values for different numbers of components. This process helps in 
finding the optimal number of components (clusters) for the GMM algorithm.'''
for i in range(1,21):
    gmm = GaussianMixture(n_components=i, random_state=0,covariance_type='diag').fit(x)
    wcss.append(gmm.aic(x))
    
#plotting the results to a graph to find the elbow:
plt.plot(range(1,21), wcss)
plt.title('AIC vs. k')
plt.xlabel('k')
plt.ylabel('AIC')
plt.show()

name = "AIC"
#applying the found k "elbow" of 8 to create kmeans classifier
# use k-means clustering to cluster data into 8 clusters
elbow_k = 8
final_gmm = GaussianMixture(n_components=elbow_k, random_state=0,covariance_type='diag').fit(x)
# use the predict() method and clusters for k=elbow_k to classify the entire iris data set.
# x = usual iris data set from iris.csv
prediction = final_gmm.predict(x)
# Match the kmean labels and the truth labels such that the number of true-positive predictions is maximized
k_labels_matched = np.empty_like(prediction)
for i in np.unique(prediction):
    # ...find and assign the best-matching truth label
    match_nums = [np.sum((prediction == i) * (y == t)) for t in np.unique(y)]
    k_labels_matched[prediction == i] = np.unique(y)[np.argmax(match_nums)]
#print confusion matrix 
print('Confusion matrix of', name, ': ')
cmatrix = np.array(confusion_matrix(y, k_labels_matched))
print(cmatrix, end='\n\n')
#print accuracy
print('Overall accuracy of', name, ': ', end=' ')
accuracy_score_ = np.trace(cmatrix) / np.sum(cmatrix)
print("No way to calculate due to the number of clusters being different from the number of classes\n\n");

#BIC
#Find the optimum k value:
wcss = []

''' For each iteration, it creates a GMM with a different number of clusters. The Bayesian Information
Criterion (BIC) value is then calculated for each GMM model using the `gmm.bic(x)` method. The BIC is a
measure of the model's quality and complexity, with lower values indicating better models. The BIC values
are then appended to the `wcss` list, which is used to store the BIC values for different numbers of 
components. This process helps in finding the optimal number of clusters for the GMM algorithm. '''
for i in range(1,21):
    gmm = GaussianMixture(n_components=i, random_state=0,covariance_type='diag').fit(x)
    wcss.append(gmm.bic(x))
    
#plotting the results to a graph to find the elbow:
plt.plot(range(1,21), wcss)
plt.title('BIC vs. k')
plt.xlabel('k')
plt.ylabel('BIC')
plt.show()

name = "BIC"
#applying the found k "elbow" of 3 to create my kmeans classifier
# use k-means clustering to cluster data into 3 clusters
elbow_k = 3
final_gmm = GaussianMixture(n_components=elbow_k, random_state=0,covariance_type='diag').fit(x)
# use the predict() method and clusters for k=elbow_k to classify the entire iris data set.
# x = usual iris data set from iris.csv
prediction = final_gmm.predict(x)
# Match the kmean labels and the truth labels such that the number of true-positive predictions is maximized
k_labels_matched = np.empty_like(prediction)
for i in np.unique(prediction):
    # ...find and assign the best-matching truth label
    match_nums = [np.sum((prediction == i) * (y == t)) for t in np.unique(y)]
    k_labels_matched[prediction == i] = np.unique(y)[np.argmax(match_nums)]
#print confusion matrix 
print('Confusion matrix of', name, ': ')
cmatrix = np.array(confusion_matrix(y, k_labels_matched))
print(cmatrix, end='\n\n')
#print accuracy
print('Overall accuracy of', name, ': ', end=' ')
accuracy_score_ = np.trace(cmatrix) / np.sum(cmatrix)
print(accuracy_score_, end='\n\n');


name = "k = 3"
#applying the found k "elbow" of 3 to create my kmeans classifier
# use k-means clustering to cluster data into 3 clusters
k = 3
final_gmm = GaussianMixture(n_components=k, random_state=0,covariance_type='diag').fit(x)
# use the predict() method and clusters for k=elbow_k to classify the entire iris data set.
# x = usual iris data set from iris.csv
prediction = final_gmm.predict(x)
# Match the kmean labels and the truth labels such that the number of true-positive predictions is maximized
k_labels_matched = np.empty_like(prediction)
for i in np.unique(prediction):
    # ...find and assign the best-matching truth label
    match_nums = [np.sum((prediction == i) * (y == t)) for t in np.unique(y)]
    k_labels_matched[prediction == i] = np.unique(y)[np.argmax(match_nums)]
#print confusion matrix 
print('Confusion matrix of', name, ': ')
cmatrix = np.array(confusion_matrix(y, k_labels_matched))
print(cmatrix, end='\n\n')
#print accuracy
print('Overall accuracy of', name, ': ', end=' ')
accuracy_score_ = np.trace(cmatrix) / np.sum(cmatrix)
print(accuracy_score_, end='\n\n');
