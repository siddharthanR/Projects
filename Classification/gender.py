from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_KNN = KNeighborsClassifier()
clf_random = RandomForestClassifier()

# Training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_KNN.fit(X, Y)
clf_random.fit(X,Y)

# Testing using the same data
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

pred_random = clf_random.predict(X)
acc_random = accuracy_score(Y, pred_random) * 100
print('Accuracy for RandomForest: {}'.format(pred_random))

# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_tree, acc_KNN, acc_random])
classifiers = {0: 'SVM', 1: 'decsion tree', 2: 'KNN', 3: 'RandomForest'}
print('Best gender classifier is {}'.format(classifiers[index]))