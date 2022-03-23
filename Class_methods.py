# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:57:04 2020

@author: W10
"""


from sklearn.datasets import load_breast_cancer 
data = load_breast_cancer()
X = data.data # Input features
y = data.target # Class label (0: Malignant, 1: Benign)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#remove the comments of the section which classifer you use#

#SVM model---------------------------------------------------------------------
#from sklearn.svm import SVC
#svm_cl = SVC(C=1e+9, gamma='scale', probability=True)
#
#svm_cl.fit(X_train, y_train) # CL is the classifier model to train 
#y_pred = svm_cl.predict(X_test) # Cancer prediction on the test set
## You need the probability of being cancer for ROC plot and AUC computation 
#y_proba = svm_cl.predict_proba(X_test)[:,1]

#bayes model-------------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB  
bayes_cl = GaussianNB(var_smoothing=1e+7)

bayes_cl.fit(X_train, y_train) # CL is the classifier model to train 
y_pred = bayes_cl.predict(X_test) # Cancer prediction on the test set
# You need the probability of being cancer for ROC plot and AUC computation 
y_proba = bayes_cl.predict_proba(X_test)[:,1]

#------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc 
# The prediction accuracy
print ( "Prediction accuracy : %.2f" % accuracy_score(y_pred,y_test ))

# Receiver Operating Cracateristic Curve (ROC)
fpr, tpr, thr = roc_curve( y_test,y_proba ) 
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Area Under ROC curve (AUC)
print ( "AUC score : %.2f" % auc( fpr, tpr ))