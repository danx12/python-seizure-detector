#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:33:21 2016

@author: danielvillarreal
"""

import numpy as np
import pylab as pl
import sklearn as sk
import scipy as sp
import glob
import pickle as pickle 


def readStoredData(name='eeg_pat22.p'):
    with open(name,'rb') as f:
        eeg = pickle.load(f)
    return eeg
    
def saveData(data,name):
    with open(name,'wb') as f:
        pickle.dump(data,f,-1)
    