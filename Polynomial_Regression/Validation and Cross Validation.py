# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:26:39 2020

@author: Hengsheng Huang
"""

## Validation and Cross Validation

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

digits = datasets.load_digits()
X = digits.data
y = digits.target


X_train, y_train, X_test, y_test = train_test_split(X, y, random_state=666)

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

param_grid = [
    {
         'weights': ['distance'],
         'n_neighbors': [i for i in range(2, 11)],
         'p': [i for i in range(1, 6)]
     }
    ]

grid_search = GridSearchCV(knn_clf, param_grid, verbose=1)
grid_search.fit(X_train, y_train)

print('best parameter:', grid_search.best_params_)
best_knn_clf = grid_search.best_estimator_
print('best score:', best_knn_clf.score(X_test, y_test))