#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 23:16:20 2016

@author: danielvillarreal
"""

import numpy as np
import pylab as pl
#import sklearn as sk
import scipy as sp
#import glob as gl
#import cPickle as pickle
from readStoredPickle import readStoredData,saveData
#from filtering import lowpassFilter
#import multiprocessing as mp
#import tabulate as tab
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss
from notify import notify

eeg_feat = readStoredData('eeg_pat22_feats.p')
X = eeg_feat['feats']
y = np.ravel(eeg_feat['labels'])

nsamples,nfeats = X.shape

classRatio = np.sum(y)/130.0e3

#%%
print 'starting OSS'
OSS = OneSidedSelection(size_ngh=51, n_seeds_S=51, n_jobs=-1)
ossx, ossy = OSS.fit_sample(X, y)

#%%
#print 'starting CC_cr'
#CC = ClusterCentroids(n_jobs=-1,ratio=classRatio)
#ccx_cr,ccy_cr = CC.fit_sample(X,y)

#%%
#print 'starting CC_a'
#CC = ClusterCentroids(n_jobs=-1,ratio='auto')
#ccx_a,ccy_a = CC.fit_sample(X,y)

#%%
#print 'starting NM3_cr'
#NM3 = NearMiss(version=3,n_jobs=-1,ratio=classRatio)
#nm3x_cr, nm3y_cr = NM3.fit_sample(X, y)
#%%
#print 'starting NM3_a'
#NM3 = NearMiss(version=3,n_jobs=-1,ratio='auto')
#nm3x_a, nm3y_a = NM3.fit_sample(X, y)

#%%
notify(True)