#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 21:08:22 2016

@author: danielvillarreal
"""

import os

def notify(mobile=False):
    if(mobile):
        os.system('curl -X POST https://maker.ifttt.com/trigger/spyder/with/key/ou3_qyIHH3IV22kLJrvLT_AJLLNR8PyUqu8dmZvIlGH')
    os.system('osascript -e \'display notification "Done in Spyder" sound name "Sound Name"\'')