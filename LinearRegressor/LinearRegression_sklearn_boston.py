# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:52:53 2020

@author: Hengsheng Huang
"""

## LinearRegression in sklearn for Boston

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]     #此处用于消除最大值不准确的误差

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

print(lin_reg.coef_)
print('intercept:', lin_reg.intercept_)
print('score:', lin_reg.score(X_test, y_test))

