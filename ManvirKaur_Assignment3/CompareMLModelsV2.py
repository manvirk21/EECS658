'''
File Name: CompareMLModelsV2.py
Purpose: 2-fold cross validatoin using the 12 different machine learning models,
then prints the confusion matrix, P, R, and F1 scores for each of the Models.
Author: Manvir Kaur
KUID: 3064194
Date: 9-28-22
Sources: Dr. Johnson's lecture notes
'''
 
# Load the necessary libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors

# Load the dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names = names)

#Naive Baysian
# Create arrays for features and classes
array = dataset.values
X = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names

#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size=0.50, random_state=1)

# Initialize Gaussian Naïve Bayes model
model = GaussianNB() #select ML model
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
# Combine predicted classes from both folds
predicted = np.concatenate([prediction1, prediction2]) #predicted classes

print("\n************************************\nNaive Baysian\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("************************************\n")




#Linear Regression
# Create arrays for features and classes
array2 = dataset.values

X = array2[:,0:4] #contains flower features (petal length, etc..)
y = array2[:,4] #contains flower names

#Encode for each class
array2 = preprocessing.LabelEncoder()
array2.fit(y)
array2.transform(y)
#Create Training and test data
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array2.transform(y), test_size=0.50, random_state=1)

#Linear Regression
LR = LinearRegression()

LR.fit(X_Fold1, y_Fold1) #first fold training
pred1 = LR.predict(X_Fold2) #First fold testing
pred1 = pred1.round() #round to integer
LR.fit(X_Fold2, y_Fold2) #second fold training
pred2 = LR.predict(X_Fold1) #second fold testing
pred2 = pred2.round() #round to integer

#evaluation:
actual = np.concatenate([y_Fold2, y_Fold1]) #actual values
predicted = np.concatenate([pred1, pred2]) #predicted values
predicted = predicted.round() #round to integer

print("\n************************************\nLinear Regression\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("************************************\n")




#Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False) #sets the degree to 2

# Create arrays for features and classes
array3 = dataset.values
X = array3[:,0:4] #contains flower features (petal length, etc..)
y = array3[:,4] #contains flower names

#Encode for each class
array3 = preprocessing.LabelEncoder()
array3.fit(y)
array3.transform(y)
#Create Training and test data
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(poly.fit_transform(X), array3.transform(y), test_size=0.50, random_state=1)

#Linear Regression
P2R = LinearRegression()
P2R.fit(X_Fold1, y_Fold1) #first fold training
pred1 = P2R.predict(X_Fold2) #First fold testing
pred1 = pred1.round() #round to integer
P2R.fit(X_Fold2, y_Fold2) #second fold training
pred2 = P2R.predict(X_Fold1) #second fold testing
pred2 = pred2.round() #round to integer

#evaluation:
actual = np.concatenate([y_Fold2, y_Fold1]) #actual values
predicted = np.concatenate([pred1, pred2]) #predicted values
predicted = predicted.round() #round to integer

print("\n************************************\nPolynomial Regression (Degree 2)\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("\n************************************\n")




#Polynomial Regression (degree 3)
poly3 = PolynomialFeatures(degree=3, include_bias=False) #sets the degree to 3
# Create arrays for features and classes
array4 = dataset.values
X = array4[:,0:4] #contains flower features (petal length, etc..)
y = array4[:,4] #contains flower names

#Encode for each class
array4 = preprocessing.LabelEncoder()
array4.fit(y)
array4.transform(y)
#Create Training and test data
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(poly.fit_transform(X), array4.transform(y), test_size=0.50, random_state=1)

#Linear Regression
P3R = LinearRegression()
P3R.fit(X_Fold1, y_Fold1) #first fold training
pred1 = P3R.predict(X_Fold2) #First fold testing
pred1 = pred1.round() #round to integer
P3R.fit(X_Fold2, y_Fold2) #second fold training
pred2 = P3R.predict(X_Fold1) #second fold testing
pred2 = pred2.round() #round to integer

#evaluation:
actual = np.concatenate([y_Fold2, y_Fold1]) #actual values
predicted = np.concatenate([pred1, pred2]) #predicted values
predicted = predicted.round() #round to integer

print("\n************************************\nPolynomial Regression (Degree 3)\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("\n************************************\n")




#kNN
from sklearn.neighbors import KNeighborsRegressor
# Create arrays for features and classes
array5 = dataset.values
X = array5[:,0:4] #contains flower features (petal length, etc..)
y = array5[:,4] #contains flower names

#Encode for each class
array5 = preprocessing.LabelEncoder()
array5.fit(y)
array5.transform(y)
#Split the data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array5.transform(y), test_size=0.50, random_state=1)

neigh = KNeighborsRegressor(n_neighbors=2) #kNN using k = 2
neigh.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = neigh.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer
neigh.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = neigh.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int

print("\n************************************\nkNN\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("************************************\n")




#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Create arrays for features and classes
array6 = dataset.values
X = array6[:,0:4] #contains flower features (petal length, etc..)
y = array6[:,4] #contains flower names

#Encode for each class
array6 = preprocessing.LabelEncoder()
array6.fit(y)
array6.transform(y)
#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array6.transform(y), test_size=0.50, random_state=1)

LDA = LinearDiscriminantAnalysis() #Uses LDA as Model
LDA.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = LDA.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer
LDA.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = LDA.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int

print("\n************************************\nLDA\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("************************************\n")




#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# Create arrays for features and classes
array7 = dataset.values
X = array7[:,0:4] #contains flower features (petal length, etc..)
y = array7[:,4] #contains flower names

#Encode for each class
array7 = preprocessing.LabelEncoder()
array7.fit(y)
array7.transform(y)
#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array7.transform(y), test_size=0.50, random_state=1)

QDA = QuadraticDiscriminantAnalysis() #Using model QDA
QDA.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = QDA.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer
QDA.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = QDA.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int
print("\n************************************\nQDA\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("************************************\n")




#SVM
from sklearn.svm import LinearSVC
# Create arrays for features and classes
array8 = dataset.values
X = array8[:,0:4] #contains flower features (petal length, etc..)
y = array8[:,4] #contains flower names

#Encode for each class
array8 = preprocessing.LabelEncoder()
array8.fit(y)
array8.transform(y)
#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array8.transform(y), test_size=0.50, random_state=1)

model = LinearSVC() #Using model SVM
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int
print("\n************************************\nSVM\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("************************************\n")




#Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Create arrays for features and classes
array9 = dataset.values
X = array9[:,0:4] #contains flower features (petal length, etc..)
y = array9[:,4] #contains flower names

#Encode for each class
array9 = preprocessing.LabelEncoder()
array9.fit(y)
array9.transform(y)
#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array9.transform(y), test_size=0.50, random_state=1)

model = DecisionTreeClassifier() #Using model DecisionTreeClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int
print("\n************************************\nDecision Tree\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("******************************************\n")




#Random Forest
from sklearn.ensemble import RandomForestClassifier

# Create arrays for features and classes
array10 = dataset.values
X = array10[:,0:4] #contains flower features (petal length, etc..)
y = array10[:,4] #contains flower names

#Encode for each class
array10 = preprocessing.LabelEncoder()
array10.fit(y)
array10.transform(y)
#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array10.transform(y), test_size=0.50, random_state=1)

model = RandomForestClassifier() #Using model RandomForestClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int
print("\n************************************\nRandom Forest\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("******************************************\n")




#Extra Tres
from sklearn.ensemble import ExtraTreesClassifier

# Create arrays for features and classes
array11 = dataset.values
X = array11[:,0:4] #contains flower features (petal length, etc..)
y = array11[:,4] #contains flower names

#Encode for each class
array11 = preprocessing.LabelEncoder()
array11.fit(y)
array11.transform(y)
#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array11.transform(y), 
test_size=0.50, random_state=1)

model = ExtraTreesClassifier() #Using model ExtraTreesClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int
print("\n************************************\nExtra Trees\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("******************************************\n")




#Neural Network
from sklearn.neural_network import MLPClassifier

# Create arrays for features and classes
array12 = dataset.values
X = array12[:,0:4] #contains flower features (petal length, etc..)
y = array12[:,4] #contains flower names

#Encode for each class
array12 = preprocessing.LabelEncoder()
array12.fit(y)
array12.transform(y)
#Split Data into 2 Folds for Training and Test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array12.transform(y), 
test_size=0.50, random_state=1)

model = MLPClassifier(max_iter=1000) #Using model MLPClassifier (up to 1000 iterations of the network)
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer

model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int

actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int
print("\n************************************\nNeral Network\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("******************************************\n")

