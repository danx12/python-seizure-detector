#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:12:09 2016

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
from entropy import sample_entropy as se
import multiprocessing as mp



def energy(window):
    return np.sum(np.power(window,2))
    
    
def rms(window):
    return np.sqrt(np.mean(np.sum(np.power(window,2))))
    
    
def linelenght(window):
    return np.sum(window[1:]-window[:-1])
    
    
def rhytmitcity(window):
    return np.std(window)/np.mean(window)
    

def sampleEntropy(window):
    a,b = se(window,2,0.2)
    return (-np.log(a[0]/b[1]))[0]

def powerfeatures(window):
    dfft = np.fft.rfft(window)
#    freqs = np.fft.rfftfreq(len(window),1.0/256.0)
    powspec = np.abs(np.power(dfft,2))
    
    bands = powspec[1:52]
    bands = np.split(bands,[5,8,16,32])
    peak = np.zeros(5)
    
    bands = np.array(map(lambda x: np.sum(x),bands))
    peak[np.argmax(bands)] = 1
    
    totalPow = np.sum(powspec[1:])
    result = np.concatenate((np.divide(bands,totalPow),np.append(peak,totalPow)))
    
    return result


def runDataset(rq,data,func,valunpack,verboseBool):
    funcName = func.func_name
    print funcName, 'Starting', str( mp.current_process().name)
    windows,samples,chan = data.shape
    result = np.zeros([windows,valunpack,chan])
#    temp = np.zeros([valunpack,chan])
    for i in range(windows):
        if((i % 10 == 0) and verboseBool):
            print 'executing',funcName,'sample:',str(i/10)
        for c in range(chan):
            result[i,:,c] = func(data[i,:,c])
    rq.put({funcName:result})
    print func.func_name, 'Exiting'
    return
    
    
if __name__ == '__main__':
    eeg = readStoredData('eeg_pat22_windows.p')
    data = eeg['data']
    labels = eeg['labels']
    #Delete eeg so we dont hold up a lot of memory
    del eeg

    fun = [energy,rms,linelenght,rhytmitcity,powerfeatures]
    unpack = [1,1,1,1,11]
    verbose = [False,False,False,False,False]

    results_queue = mp.Queue()
    jobs = []
    for f,u,v in zip(fun,unpack,verbose):
        p = mp.Process(target=runDataset, args=(results_queue,data,f,u,v,))
        jobs.append(p)
        p.start()
        
#    for p in jobs:
#        p.join()
    
    results = [results_queue.get() for j in jobs]
    feats = {}
    for f in results:
        n = f.keys()[0]
        feats[n] = f[n]
    
    

#    jobs = []
#    manager = mp.Manager()
#    return_dict = manager.dict()

