# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:41:11 2020

@author: samue
"""
## MNIST

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

mnist = fetch_openml('mnist_784')


X, y = mnist['data'], mnist['target']
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(X[60000:], dtype=float)


# KNN
#knn_clf = KNeighborsClassifier()
#knn_clf.fit(X_train, y_train)
#print('KNN:', knn_clf.score(X_test, y_test))


# PCA
pca = PCA(0.90)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

knn_clf2 = KNeighborsClassifier()
knn_clf2.fit(X_train_reduction, y_train)

print('with PAC:', knn_clf2.score(X_test_reduction, y_test))