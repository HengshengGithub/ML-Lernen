# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:53:44 2020

@author: Hengsheng Huang
"""


## SGD in sklearn

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

m = 100000

x = np.random.normal(size=m)
X = x.reshape(-1, 1)
y = 4.* x + 3. + np.random.normal(0, 3, size=m)

# test-gruppe-creative
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)

# normalization
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)


#SGD
sgd_reg = SGDRegressor()
sgd_reg.fit(X_train_standard, y_train)
print(sgd_reg.score(X_test_standard, y_test))
