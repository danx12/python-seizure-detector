#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:24:20 2016

@author: danielvillarreal
"""

import numpy as np
import pylab as pl
import scipy as sp
from readStoredPickle import readStoredData,saveData
from entropy import sample_entropy as se
import multiprocessing as mp
import tabulate as tab
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

#%%
eeg_train = readStoredData('cc_automatic.p')
X = eeg_train['feats']
y = eeg_train['labels']

eeg_alt = readStoredData('eeg_pat22_feats.p')
X_alt = eeg_alt['feats']
y_alt = np.ravel(eeg_alt['labels'])


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=41)

#tuned_parameters = [{'kernel':['linear'],'C':[1,10,100,1000]}
#                    ,
#                    {'kernel':['rbf'],'gamma':[1e0,1e-1,1e-2,1e-3,1e-4,1e-5],'C':[1,10,100,1000]}
#                    ]

tuned_parameters = [{'C':[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,0.5,1,10,100,1000,2000,3000,4000,5000,7000,9000,10000]}]
                    
#scores = ['precision_macro','recall_macro','roc_auc']
#scores = ['f1','f1_macro']
scores = ['roc_auc']

#f1

#%%

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(LinearSVC(class_weight='balanced',random_state=41), tuned_parameters, cv=10,scoring='%s' % score,
                       n_jobs=-1,verbose=0)
    clf.fit(X_train, np.ravel(y_train))

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Confusion matrix")
    print(confusion_matrix(y_true,y_pred))
    print()
    cf = np.ravel(clf.best_estimator_.coef_)
