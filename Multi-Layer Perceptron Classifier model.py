# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:48:12 2016

@author: veda
"""

import sklearn
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print (cancer['DESCR'])
cancer['data'].shape

x = cancer['data']
y = cancer['target']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



#he Multi-Layer Perceptron Classifier model) from the neural_network library of SciKit-Learn!
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)

Predictions and Evaluation
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

