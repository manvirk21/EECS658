'''
File Name: CompareFeatureSelectionMethods.py
Author: Manvir Kaur
KUID: 3064194
Date: 10-12-23
Sources: Dr. Johnson's lecture notes
'''

# Load the necessary libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import preprocessing
import pandas as pd


# Load the dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names = names)

#Part 1
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

#Initialize model
model = DecisionTreeClassifier() #Using model DecisionTreeClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
# Combine predicted classes from both folds
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int

print("\n************************************\nPart 1\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("************************************\n")



#Part 2
from sklearn.decomposition import PCA

# Create arrays for features and classes
array10 = dataset.values
x = array10[:,0:4] #contains flower features (petal length, etc..)
y = array10[:,4] #contains flower names

# Create PCA instance
pca = PCA(n_components = 4)
# Perform PCA
pca.fit(x)

# Get eigenvectors and eigenvalues
eigenvectors = pca.components_
eigenvalues = pca.explained_variance_

# Transform data
principleComponents = pca.transform(x)

# Calculate PoVs
sumvariance = np.cumsum(eigenvalues)
sumvariance /= sumvariance[-1]

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = list(zip(eigenvalues, eigenvectors))

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

# Transform data (x) to Z
W = eigen_pairs[0][1].reshape(4, 1)
Z = principleComponents.dot(W)

print(W)

# Encode for each class
array10 = preprocessing.LabelEncoder()
array10.fit(y)
array10.transform(y)

#Split data into 2 folds for training and test
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(x, array10.transform(y), test_size=0.50, random_state=1)

model = DecisionTreeClassifier() #Using model DecisionTreeClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
# Combine predicted classes from both folds
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int
print("\n\n************************************\nPart 2\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("************************************\n")



########Extra Trees################
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
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, array11.transform(y), test_size=0.50, random_state=1)

model = ExtraTreesClassifier() #Using model ExtraTreesClassifier
model.fit(X_Fold1, y_Fold1) #first fold training
prediction1 = model.predict(X_Fold2) #first fold testing
prediction1 = prediction1.round() #round to integer
model.fit(X_Fold2, y_Fold2) #second fold training
prediction2 = model.predict(X_Fold1) #second fold testing
prediction2 = prediction2.round() # round to int

# Combine actual classes from both folds
actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
# Combine predicted classes from both folds
predicted = np.concatenate([prediction1, prediction2]) #predicted classes
predicted = predicted.round() #round to int
print("\n************************************\nExtra Trees\n************************************")
print(accuracy_score(actual, predicted)) #accuracy
print(confusion_matrix(actual, predicted)) #confusion matrix
print(classification_report(actual, predicted)) #P, R, & F1
print("************************************\n")


#Part 3
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import random
def evaluate_subset(X, y, feature_indices):
    X_subset = X[:, feature_indices]
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_subset, y, test_size=0.50, random_state=1)

    model = ExtraTreesClassifier()
    model.fit(X_Fold1, y_Fold1)
    prediction1 = model.predict(X_Fold2)
    prediction1 = prediction1.round()
    model.fit(X_Fold2, y_Fold2)
    prediction2 = model.predict(X_Fold1)
    prediction2 = prediction2.round()

    actual = np.concatenate([y_Fold2, y_Fold1])
    predicted = np.concatenate([prediction1, prediction2])
    predicted = predicted.round()

    return accuracy_score(actual, predicted)

# Define the simulated annealing algorithm
def simulated_annealing(X, y, initial_solution, iterations, perturb_percentage, restarts):
    current_solution = initial_solution
    current_score = evaluate_subset(X, y, current_solution)

    best_solution = current_solution
    best_score = current_score

    for iteration in range(iterations):
        new_solution = current_solution.copy()
        
        # Perturb the solution
        for _ in range(int(len(new_solution) * perturb_percentage)):
            index = random.randint(0, len(new_solution) - 1)
            new_solution[index] = 1 - new_solution[index]

        new_score = evaluate_subset(X, y, new_solution)

        # Determine whether to accept the new solution
        if new_score > current_score or random.random() < 1:
            current_solution = new_solution
            current_score = new_score

        # Check for improvements and perform restarts
        if current_score > best_score:
            best_solution = current_solution
            best_score = current_score
        elif iteration % restarts == 0:
            current_solution = best_solution
            current_score = best_score

        # Print results for each iteration
        print("Iteration", iteration + 1)
        print("Subset of features:", current_solution)
        print("Accuracy:", current_score)
        print("Pr[accept]:", random.random())
        print("Random Uniform:", random.uniform(0, 1))
        print("Status: Improved" if current_score > best_score else "Accepted" if new_score > current_score else "Discarded" if random.random() >= 1 else "Restart")
        print("************************************")

# Create arrays for features and classes
array12 = dataset.values
X = array12[:, 0:4]
y = array12[:, 4]

# Encode for each class
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
y_encoded = label_encoder.transform(y)

# Set the parameters for simulated annealing
initial_solution = [1, 1, 1, 1, 1, 1, 1, 1]  # Start with all features
iterations = 100
perturb_percentage = 0.05  # 1-5% of 8 features
restarts = 10

# Run the simulated annealing algorithm
simulated_annealing(X, y_encoded, initial_solution, iterations, perturb_percentage, restarts)



#Part 4
# Define the genetic algorithm
def genetic_algorithm(X, y, population_size, generations):
    num_features = X.shape[1]

    # Initialize the population
    population = []
    for _ in range(population_size):
        individual = [random.randint(0, 1) for _ in range(num_features)]
        population.append(individual)

    for generation in range(generations):
        # Evaluate the fitness of each individual in the population
        fitness = [evaluate_subset(X, y, individual) for individual in population]

        # Select the top performing individuals
        num_parents = int(0.2 * population_size)
        parents = np.argsort(fitness)[-num_parents:]

        # Create a new population by recombination and mutation
        new_population = []

        for _ in range(population_size - num_parents):
            parent1 = population[random.choice(parents)]
            parent2 = population[random.choice(parents)]

            # Perform one-point crossover
            crossover_point = random.randint(1, num_features - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]

            # Perform mutation
            mutation_rate = 0.1
            for i in range(num_features):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]

            new_population.append(child)

        # Replace the old population with the new population
        population[:population_size - num_parents] = new_population
        best_individual = population[np.argmax(fitness)]

        # Print results for each generation
        print("Generation", generation + 1)
        print("Subset of features:", best_individual)
        print("Accuracy:", max(fitness))
        print("************************************")

# Create arrays for features and classes
array12 = dataset.values
X = array12[:, 0:4]
y = array12[:, 4]

# Encode for each class
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
y_encoded = label_encoder.transform(y)

# Set the parameters for the genetic algorithm
population_size = 20
generations = 50

# Run the genetic algorithm
genetic_algorithm(X, y_encoded, population_size, generations)
