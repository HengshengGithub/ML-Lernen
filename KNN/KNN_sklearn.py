# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:03:30 2020

@author: samue
"""
# sklearn mit knn und test_split fuer datasets_digits

import numpy as np
from sklearn import datasets

digits = datasets.load_digits()

X = digits.data
y = digits.target



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test, y_test))

#y_predict = knn_clf.predict(X_test)

#from sklearn.metrics import accuracy_score

#print(accuracy_score(y_test, y_predict))

