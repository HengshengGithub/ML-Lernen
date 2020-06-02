# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:42:48 2020

@author: Hengsheng Huang
"""

## Logistic-Regression in sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import plot_decision_boundary

# create the daten
np.random.seed(666)
X = np.random.normal(0, 1, size=(200, 2))
y = np.array((X[:,0]**2+X[:,1])<1.5, dtype='int')
for _ in range(20):
    y[np.random.randint(200)] = 1

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# Linear-LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print('Linear_train_score:', log_reg.score(X_train, y_train))
print('Linear_test_score:', log_reg.score(X_test, y_test))

def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X_train, y_train)

print('PolynomialLogReg_train_score:', poly_log_reg.score(X_train, y_train))
print('PolynomialLogReg_train_score:', poly_log_reg.score(X_test, y_test))
    
plot_decision_boundary.plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

