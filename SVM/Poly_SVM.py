# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:56:00 2020

@author: Hengsheng Huang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import plot_decision_boundary as pdb


X, y = datasets.make_moons()
X, y = datasets.make_moons(noise=0.15, random_state=666)
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# SVM mit poly
def PolynomialKernelSVC(degree, C=1.0):
    return Pipeline([
            ('std_scaler', StandardScaler()),
            ('kernelSVC', SVC(kernel='poly',degree=degree, C=C))
        ])

poly_kernel_svc = PolynomialKernelSVC(degree=3)
poly_kernel_svc.fit(X, y)

# Visiualisierung
pdb.plot_decision_boundary(poly_kernel_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# SVM mit RBF
def RBFKernelSVC(gamma):
    return Pipeline([
            ('std_scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', gamma=gamma))
        ])

svc = RBFKernelSVC(gamma = 1)
svc.fit(X, y)

pdb.plot_decision_boundary(svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()