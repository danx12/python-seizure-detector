#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:53:36 2016

@author: danielvillarreal
"""

import numpy as np
import pylab as pl
import sklearn as sk
import scipy as sp
import glob
import cPickle as pickle
from readStoredPickle import readStoredData,saveData
from filtering import lowpassFilter


eeg = readStoredData('eeg_pat22.p')
data = eeg['data']
labels = eeg['labels']
data = lowpassFilter(data)


def scaledata(oneChanData,minVal,maxVal):
    data = oneChanData.astype(float)
    newRange = maxVal - minVal
    oldMax = np.max(data)
    oldRange = oldMax - np.min(data)
    
    a = newRange/oldRange
    data = a*(data - oldMax) + maxVal
    return data.astype('uint16')


def scaleDataset(dataset,minVal,maxVal):
    rows,chan = dataset.shape
    result = np.zeros(dataset.shape)
    for c in range(0,chan):
        result[:,c] = scaledata(dataset[:,c],minVal,maxVal)
        
    return result.astype('uint16')
    
    
#so this can be simmilar to 12 bit ADC
data = scaleDataset(data,0.0,float(2**12)-1.0)

def windowsOfData(data,wsize,samplingrate):
    rows,channels = data.shape
    samplesPerWindow = rows / float(samplingrate)
    samplesPerWindow /= wsize
    windows = np.zeros([int(samplesPerWindow),int(samplingrate),channels])
    windows = np.array(np.split(data,samplesPerWindow))
    return windows,samplesPerWindow

#convert data and labels into windows 
data,samplesPerWindow = windowsOfData(data,1,256)
labels = np.sum(np.split(labels,samplesPerWindow),axis=1)
labels = np.where(labels > 0,1,0).astype('uint16')

windowedEeg = {'data':data,'labels':labels}
saveData(windowedEeg,'pat22_eeg_windows.p') 
    