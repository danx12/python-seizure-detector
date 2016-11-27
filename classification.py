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
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

#%%
eeg_feat = readStoredData('cc_automatic.p')
X = eeg_feat['feats']
y = eeg_feat['labels']

eeg_alt = readStoredData('eeg_pat22_feats.p')
X_alt = eeg_alt['feats']
y_alt = np.ravel(eeg_alt['labels'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=41)

tuned_parameters = [{'kernel':['rbf'],'gamma':[2e-15,2e-10,2e-5,2e0,2e1,2e3],'C':[1,10,100,1000]}]

#tuned_parameters = [{'kernel':['rnf'],'C':[1,10,100]}]
                    
scores = ['precision_macro','recall_macro','roc_auc']

#f1

#%%

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(class_weight='balanced',cache_size=700,random_state=12), tuned_parameters, cv=3,scoring='%s' % score,
                       n_jobs=-1,verbose=4)
    clf.fit(X_train, y_train)

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
    print()