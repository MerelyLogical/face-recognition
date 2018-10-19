#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:23:56 2018

@author: edward
"""

import numpy as np

k = np.array([[1,-3,3],[3,-5,3],[6,-6,4]])

val,vec = np.linalg.eig(k)

bigmatrix = np.empty((len(val), 2))
for i in range(len(val)):
    bigmatrix[i] = ([abs(val[i]), vec[:,i]])

#bigmatrix = bigmatrix[bigmatrix[:,0].argsort()]
