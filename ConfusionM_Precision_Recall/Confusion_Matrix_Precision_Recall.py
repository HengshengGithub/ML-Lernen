# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:46:10 2020

@author: Hengsheng Huang
"""

#Confusion-Matrix-Precision and Recall
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print('LogReg score:', log_reg.score(X_test, y_test))
decision_scores = log_reg.decision_function(X_test)
y_log_predict = log_reg.predict(X_test)


# confusion_matrix
print('Confusion matrix:', confusion_matrix(y_test, y_log_predict))

# precision_score
print('Precision score:', precision_score(y_test, y_log_predict))

# recall_score
print('Recall score:', recall_score(y_test, y_log_predict))

# F1_score
print('F1 score:', f1_score(y_test, y_log_predict))

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, decision_scores)
plt.plot(thresholds, precisions[:-1])
plt.plot(thresholds, recalls[:-1])
plt.show()

plt.plot(precisions, recalls)
plt.show()

# ROC
fprs, tprs, thresholds = roc_curve(y_test, decision_scores)
plt.plot(fprs, tprs)
plt.show()

# ROC_AUC
print('ROC AUC:', roc_auc_score(y_test, decision_scores))


