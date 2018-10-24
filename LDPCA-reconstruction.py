#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:55:07 2018

@author: zw4315, zw4315
"""

#Q1 Application of Eigen faces 
#a) reconstruction while varying the number of PCA learnt using LD PCA

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

M = 100

LDeivec = LDeivec.transpose()
LDindexEigenvalue = np.argsort(LDeival)

LDpcaEigenval = LDeival[LDindexEigenvalue[-M:]]
LDpcaEigenvec = LDeivec[LDindexEigenvalue[-M:]]



LDreeigenvec = np.zeros(2576)
for i in range(0, M):
    LDvecdir = np.dot(phi_matrix, LDpcaEigenvec[i,:])      # A*v = u
    LDnormed = LDvecdir/np.linalg.norm(LDvecdir) 
    LDreeigenvec = np.column_stack((LDreeigenvec ,LDnormed))
    
    

bestLDpcaEigenvec = LDreeigenvec.transpose()[::-1][:-1,:]   
bestLDpcaEigenval = LDpcaEigenval[::-1]        # print the eigenface with high energy first


#reconstruction of the image
mean_face = np.mean(data_train, axis=1) 
for pic_idx in range (0, 10):
    LDai = np.dot(phi_matrix[:,pic_idx].transpose(), bestLDpcaEigenvec.transpose()[:, 0])
    LDsum = mean_face +  np.dot(LDai,bestLDpcaEigenvec.transpose()[:, 0])
    
    for i in range(1,M):    
        LDai = np.dot(phi_matrix[:,pic_idx].transpose(), bestLDpcaEigenvec.transpose()[:, i])
        LDsum = LDsum + np.dot(LDai,bestLDpcaEigenvec.transpose()[:, i])
    
    original_pic =  np.swapaxes( np.reshape( np.array(data_train[:,pic_idx]), (46, 56)), 0, 1)
    LDreconstruct_pic =  np.swapaxes( np.reshape( np.array(LDsum), (46, 56)), 0, 1)
   
    
    plt.subplot(5,4,2*pic_idx+1)
    plt.axis('off')
    plt.imshow(abs(original_pic ),cmap='gray')
    
    
    plt.subplot(5,4,2*pic_idx+2)
    plt.axis('off')
    plt.imshow(abs(LDreconstruct_pic),cmap='gray')

