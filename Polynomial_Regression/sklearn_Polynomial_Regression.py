# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:42:16 2020

@author: Hengsheng Huang
"""


## Polynomial-Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, 100)

poly = PolynomialFeatures()
poly.fit(X)
X2 = poly.transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X2, y)
y_predict2 = lin_reg.predict(X2)

plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()


# Pipeline
poly_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('Std_scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
    ])

poly_reg.fit(X, y)
y_predict = poly_reg.predict(X)

plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
plt.show()