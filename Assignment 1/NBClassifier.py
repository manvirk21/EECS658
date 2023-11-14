'''
Name: EECS 658 Assignment 1
Author: Manvir Kaur
KUID: 3064194
Date: 08/31/2023
Purpose: A Python program to demonstrate the Naïve Bayesian Classifier using
Gaussian Naïve Bayes model on the Iris dataset with 2-fold cross-validation.
Input: iris.csv could be the only thing counted as an input
Output: The program prints out accuracy, confusion matrix, and classification report.
Sources: Dr. Johnson lecture slides
'''

# Load the necessary libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Load the dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names = names)


# Create arrays for features and classes
array = dataset.values
X = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names

#Split the data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size=0.50, random_state=1)

# Initialize Gaussian Naïve Bayes model
model = GaussianNB() #select ML model
model.fit(X_Fold1, y_Fold1) #first fold training
pred1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
pred2 = model.predict(X_Fold1) #second fold testing

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1])

# Combine predicted classes from both folds
predicted = np.concatenate([pred1, pred2])

# Calculate and print accuracy
accuracy = accuracy_score(actual, predicted)
print("The accuracy is", accuracy)

# Print confusion matrix
print("\nConfusion Matrix: \n", confusion_matrix(actual, predicted))

# Print classification report (P, R, & F1 scores)
print(classification_report(actual, predicted)) #P, R, & F1
