#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:44:37 2018

@author: zw4215, zw4315
"""

# Q1 Eigen faces
# b) using the low-dimensional computation of eigenface in comparison to a)

import numpy as np
import matplotlib.pyplot as plt

#load the variables from files
data_train = np.load("data_train.npy");
label_train = np.load("label_train.npy");
data_test = np.load("data_test.npy");
label_test = np.load("label_test.npy");
phi_matrix = np.load("phi_matrix.npy");
eival = np.load("eival.npy");
eivec = np.load("eivec.npy");
LDeival = np.load("LDeival.npy");
LDeivec = np.load("LDeivec.npy");

M = 20

eivec = eivec.transpose()
LDeivec = LDeivec.transpose()

indexEigenvalue = np.argsort(eival)
LDindexEigenvalue = np.argsort(LDeival)


pcaEigenval = eival[indexEigenvalue[-M:]]
LDpcaEigenval = LDeival[LDindexEigenvalue[-M:]]

pcaEigenvec = eivec[indexEigenvalue[-M:]]
LDpcaEigenvec = LDeivec[LDindexEigenvalue[-M:]]

EigenVal_difference =  pcaEigenval - LDpcaEigenval   # success, so low dimensional computation gets the same eigen value

EigenVec_difference = np.zeros(2576)
LDreeigenvec = np.zeros(2576)

for i in range(0, M):
    LDvecdir = np.dot(phi_matrix, LDpcaEigenvec[i,:])
    LDnormed = LDvecdir/np.linalg.norm(LDvecdir)
    vecdir = pcaEigenvec[i,:]
    normedvec = vecdir/np.linalg.norm(vecdir)
    
    LDreeigenvec = np.column_stack((LDreeigenvec ,LDnormed))
    EigenVec_difference =  np.column_stack((EigenVec_difference, (abs(normedvec) - abs(LDnormed))))
    

bestpcaEigenvec = pcaEigenvec[::-1]
bestLDpcaEigenvec = LDreeigenvec.transpose()[::-1][:-1,:]   # print the eigenface with high energy first

for i in range(0, 20):
    pic = np.swapaxes( np.reshape( np.array(bestLDpcaEigenvec[i,:]), (46, 56) ), 0, 1)
    plt.subplot(4,5,i+1)
    plt.axis('off')
    plt.imshow(abs(pic), cmap='gray')

