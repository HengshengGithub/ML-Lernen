# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:06:47 2020

@author: samue
"""
# Mit Grid Search

from sklearn import datasets
from sklearn.model_selection import GridSearchCV

digits = datasets.load_digits()

X = digits.data
y = digits.target



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=233)

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_clf.fit(X_train, y_train)
#print(knn_clf.score(X_test, y_test))


param_grid =[
    {
         'weights': ['uniform'],
         'n_neighbors': [i for i in range (1, 11)]
     },
    {
         'weights': ['distance'],
         'n_neighbors': [i for i in range(1, 11)],
         'p': [i for i in range(1, 6)]
     }
    ]

knn_clf = KNeighborsClassifier()


grid_search = GridSearchCV(knn_clf, param_grid)

grid_search.fit(X_train, y_train)

print(grid_search.best_estimator_)

print('best score:', grid_search.best_score_)