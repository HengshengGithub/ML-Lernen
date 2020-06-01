# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:13:45 2020

@author: Hengsheng Huang
"""

## sklearn SGD for boston

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# normalization
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)


#SGD
sgd_reg = SGDRegressor(max_iter=500)
sgd_reg.fit(X_train_standard, y_train)
print(sgd_reg.score(X_test_standard, y_test))