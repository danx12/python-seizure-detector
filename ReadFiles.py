#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:43:49 2016

@author: danielvillarreal
"""

import numpy as np
import pylab as pl
import sklearn as sk
import scipy as sp
import glob
import cPickle as pickle


eventInfo = np.genfromtxt('events.csv',delimiter=',',dtype='int')
numEvents = eventInfo.shape[0]

data = []
labels = []

dirpath = '../surf30/pat_22602/adm_226102/rec_22600102'
for i,path in enumerate(glob.glob('%s/22600102_*.data' % dirpath)):
    print("Processing batch: " + str(i))
    batch = np.fromfile(path,dtype='int16',count=-1)
    rows = len(batch)/29
    batch = (batch.reshape([rows,29]).astype('float16') * 0.165)[:,[17,2,4]]
    data.append(batch)
    lb = np.zeros([rows,1])
    for f,start,end in eventInfo:
        if(f == i):
            lb[start:end+1,:] = 1
    labels.append(lb)
        
eeg = {'data':np.concatenate(data,axis=0),'labels':np.concatenate(labels,axis=0)};
       
       
#pickle.dump(eeg,open( 'eeg_pat22.pickle', 'wb' ))
#pickle.load(eeg,open( 'eeg_pat22.pickle', 'wb' ))