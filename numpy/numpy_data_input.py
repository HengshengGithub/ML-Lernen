# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:23:04 2020

@author: samue
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
plt.scatter(X[:,0], X[:,1])
plt.show()

y = iris.target
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='o') #使用了fancy-Indexing， index为0，取出第0个第0列和第0个第一列两组数据
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='+')
plt.scatter(X[y==2, 0], X[y==2, 1], color='green', marker='x')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

Z = iris.data[:, 2:]
plt.scatter(Z[:,0], X[:,1])
plt.show()

plt.scatter(Z[y==0, 0], Z[y==0, 1], color='red', marker='o') #使用了fancy-Indexing， index为0，取出第0个第0列和第0个第一列两组数据
plt.scatter(Z[y==1, 0], Z[y==1, 1], color='blue', marker='+')
plt.scatter(Z[y==2, 0], Z[y==2, 1], color='green', marker='x')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()