'''
File Name: imbalanced.py
Description: uses a variety of over and undersampling methods to train a neural_network model to use the imbalanced iris set of data.
Author: Manvir Kaur
KUID: 3064194
Date: 10-26-23
Inputs: imbalanced iris.csv
Outputs: prints out confusion matrix and accuracy for different methods
Sources: Dr. Johnson's lecture notes, debugging with ChatGPT
'''
 
# Load the necessary libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np
from imblearn.over_sampling import RandomOverSampler 
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import ADASYN 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids 
from imblearn.under_sampling import TomekLinks 


# Load the dataset, skip the first row with column names
url = "imbalanced iris.csv"
dataset = read_csv(url, skiprows=[0], names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])


#Part 1
# Create arrays for features and classes
array12 = dataset.values
X = array12[:,0:4] #contains flower features (petal length, etc..)
y = array12[:,4] #contains flower names

#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size=0.50, random_state=1)

model = MLPClassifier(max_iter=1000) #Using model MLPClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
print("\n************************************\nPart 1\n************************************")
print("Accuracy: ", end = ''); 
print(accuracy_score(actual, predicted)) #accuracy
print("Confusion Matrix: ");
print(confusion_matrix(actual, predicted)) #confusion matrix
print("Class Balance Accuracy: ", end = '');
print(2.84/3); #class balance accuracy
print("Balanced Accuracy: ", end = '');
print((1+(90/93)+1)/3); #balance accuracy
print("balanced_accuracy_score: ", end = '');
print(balanced_accuracy_score(actual,predicted)) #balance accuracy
print("************************************\n")


#Part 2
# Create arrays for features and classes
array12 = dataset.values
X = array12[:,0:4] #contains flower features (petal length, etc..)
y = array12[:,4] #contains flower names

ros = RandomOverSampler(random_state=50)
X_res, y_res = ros.fit_resample(X, y)

#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size=0.50, random_state=1)

model = MLPClassifier(max_iter=1000) #Using model MLPClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
# Combine predicted classes from both folds
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
print("\n\n************************************\nPart 2\n************************************")
print("\n************************************\nRandom Oversampling\n************************************")
print("Accuracy: ", end = ''); 
print(accuracy_score(actual, predicted)) #accuracy
print("Confusion Matrix: ");
print(confusion_matrix(actual, predicted)) #confusion matrix
print("************************************\n")


sm = SMOTE(random_state=50)
X_res, y_res = sm.fit_resample(X, y)
#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size=0.50, random_state=1)

model = MLPClassifier(max_iter=1000) #Using model MLPClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
# Combine predicted classes from both folds
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
print("\n************************************\nSMOTE Oversampling\n************************************")
print("Accuracy: ", end = ''); 
print(accuracy_score(actual, predicted)) #accuracy
print("Confusion Matrix: ");
print(confusion_matrix(actual, predicted)) #confusion matrix
print("************************************\n")


ada = ADASYN(random_state=50, sampling_strategy='minority')
X_res, y_res = ada.fit_resample(X, y)

#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size=0.50, random_state=1)

model = MLPClassifier(max_iter=1000) #Using model MLPClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
# Combine predicted classes from both folds
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
print("\n************************************\nADASYN Oversampling\n************************************")
print("Accuracy: ", end = ''); 
print(accuracy_score(actual, predicted)) #accuracy
print("Confusion Matrix: ");
print(confusion_matrix(actual, predicted)) #confusion matrix
print("***********************************\n")


#Part 3
#Create Arrays for Features and Classes
array12 = dataset.values
X = array12[:,0:4] #contains flower features (petal length, etc..)
y = array12[:,4] #contains flower names

rus = RandomUnderSampler(random_state=30)
X_res, y_res = rus.fit_resample(X, y)

#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size=0.50, random_state=1)

model = MLPClassifier(max_iter=1000) #Using model MLPClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
# Combine predicted classes from both folds
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
print("\n************************************\nPart 3\n************************************")
print("\n************************************\nRandom Undersampling\n************************************")
print("Accuracy: ", end = ''); 
print(accuracy_score(actual, predicted)) #accuracy
print("Confusion Matrix: ");
print(confusion_matrix(actual, predicted)) #confusion matrix
print("************************************\n")


cc = ClusterCentroids(random_state=30)
X_res, y_res = cc.fit_resample(X, y)

#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size=0.50, random_state=1)

model = MLPClassifier(max_iter=1000) #Using model MLPClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
# Combine predicted classes from both folds
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
print("\n***********************************\nCluster Undersampling\n***********************************")
print("Accuracy: ", end = ''); 
print(accuracy_score(actual, predicted)) #accuracy
print("Confusion Matrix: ");
print(confusion_matrix(actual, predicted)) #confusion matrix
print("***********************************\n")


tl = TomekLinks()
X_res, y_res = tl.fit_resample(X, y)

#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_res, y_res, test_size=0.50, random_state=1)

model = MLPClassifier(max_iter=1000) #Using model MLPClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
# Combine predicted classes from both folds
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
print("\n**********************************\nTomekLinks Undersampling\n**********************************")
print("Accuracy: ", end = ''); 
print(accuracy_score(actual, predicted)) #accuracy
print("Confusion Matrix: ");
print(confusion_matrix(actual, predicted)) #confusion matrix
print("***********************************\n")

