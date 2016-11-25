e#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:12:09 2016

@author: danielvillarreal
"""

import numpy as np
#import pylab as pl
#import sklearn as sk
import scipy as sp
#import glob as gl
#import cPickle as pickle
from readStoredPickle import readStoredData,saveData
#from filtering import lowpassFilter
from entropy import sample_entropy as se
import multiprocessing as mp
#import tabulate as tab

#%%

def mean(window):
    return np.mean(window).astype('int32')

    
def skewness(window):
    return sp.stats.skew(window)
    
    
def kurtosis(window):
    return sp.stats.kurtosis(window)

    
def energy(window):
    return np.sum(np.power(window,2).astype('int32')).astype('int32')
    
#allows floating point arithmetic
def std(window):
    return np.std(window)
    
    
def rms(window):
    return np.sqrt(np.mean(np.sum(np.power(window,2).astype('int32'))).astype('int32')).astype('int32')
    

def linelenght(window):
    return np.abs(np.sum(window[1:]-window[:-1])).astype('int32')
    
#allows floating point arithmetic   
def rhytmitcity(window):
    return np.std(window)/np.mean(window)
    

def sampleEntropy(window):
    a,b = se(window,2,0.2)
    return (-np.log(a[0]/b[1]))[0]

def powerfeatures(window):
    dfft = np.fft.rfft(window)
#    freqs = np.fft.rfftfreq(len(window),1.0/256.0)
    powspec = np.abs(np.real(dfft).astype('int32') + np.imag(dfft).astype('int32')).astype('int32')
    
    bands = powspec[1:52]
    bands = np.split(bands,[5,8,16,32])
    peak = np.zeros(5,dtype='int32')
    
    bands = np.array(map(lambda x: np.sum(x).astype('int32'),bands))
    peak[np.argmax(bands)] = 1.0
    
#         filter out the dc band
    totalPow = np.sum(powspec[1:])
   
    if(totalPow == 0.0):
        result = np.zeros([1,11])
    else:
        result = np.concatenate(((bands,np.append(peak,totalPow)))).astype('int32')
    
    
    return result.astype('int32')


def runDataset(rq,data,func,valunpack,verboseBool):
    funcName = func.func_name
    print funcName, 'Starting', str( mp.current_process().name)
    windows,samples,chan = data.shape
    result = np.zeros([windows,valunpack,chan])
#    temp = np.zeros([valunpack,chan])
    for i in range(windows):
        if((i % 10 == 0) and verboseBool):
            print 'executing',funcName,'sample:',str(i/10)
        for c in range(0,chan):
            result[i,:,c] = func(data[i,:,c])
    rq.put({funcName:np.reshape(result,[windows,valunpack*chan])})
    print func.func_name, 'Exiting'
    return
    
    
if __name__ == '__main__':
    eeg = readStoredData('eeg_pat22_windows.p')
    data = eeg['data']
    labels = eeg['labels']
    #Delete eeg so we dont hold up a lot of memory
    del eeg

    fun = [energy,rms,linelenght,rhytmitcity,powerfeatures,mean,std]
    unpack = [1,1,1,1,11,1,1]
    verbose = [False,False,False,False,False,False,False]

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
        feats[n] = (f[n])
#%%    
    globalMax = 2**31 
    featScale = {'mean':2**12-1,'energy':(2**31),'std':2049,'rms':2**12,
    'linelenght':2**24,'rhytmitcity':np.sqrt(256-1).astype('int32'),
    'powerfeatures':np.array([(2**20)-1, 1, (2**20)-1])}
    
#   compute scaling factors:
    for key in featScale.keys():
        if(key != 'powerfeatures'):
            featScale[key] = globalMax/featScale[key]
        else:
            for i in range(0,len(featScale[key])):
                featScale[key][i] = globalMax/featScale[key][i]
                
#%%

#computer scaled features and vectorized
    nsamples = data.shape[0]
    validFeats = []
    order=[]
#    count = 0
    for key,value in feats.iteritems():
        order.append(key)
        if(key != 'powerfeatures'):
            validFeats.append((value*featScale[key]).astype('int32'))
        else:
            bands = (value[:,0:15]*featScale[key][0]).astype('int32')
            peak = (value[:,15:30]*featScale[key][1])
            total = value[:,30:].astype('int32')*featScale[key][2]
            validFeats.append(bands)
            validFeats.append(peak)
            validFeats.append(total)
            
    validFeats = np.concatenate(validFeats,axis=1)
            
            
            
#            count +=3
#        else:
#            scaledFeats[:,count:count+15] = value[0:5].reshape([nsamples,3*5])*maxFeature[key][0]
#            count += 15
#            scaledFeats[:,count:count+15] = value[5:11].reshape([nsamples,3*5])*maxFeature[key][1]
#            count+=15
#            scaledFeats[:,count:count+3] = value[11].reshape([nsamples,3])*maxFeature[key][-1]
#            count+=3

#    jobs = []
#    manager = mp.Manager()
#    return_dict = manager.dict()

