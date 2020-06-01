# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:45:28 2020

@author: Hengsheng Huang
"""


## Sklearn for PCA with digits

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# KNN
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

print('Ergebnisse von KNN:', knn_clf.score(X_test, y_test))


# PCA with 2D
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

knn_clf2 = KNeighborsClassifier()
knn_clf2.fit(X_train_reduction, y_train)

print('Ergebnisse mit PCA:', knn_clf2.score(X_test_reduction, y_test))


# PCA with nD
pca = PCA(0.95)
pca.fit(X_train)
X_train_reduction2 = pca.transform(X_train)
X_test_reduction2 = pca.transform(X_test)

knn_clf3 = KNeighborsClassifier()
knn_clf3.fit(X_train_reduction2, y_train)

print('Zahl von n:', pca.n_components_)
print('Ergebnisse mit nD PCA:', knn_clf3.score(X_test_reduction2, y_test))


