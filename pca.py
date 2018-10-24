#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:07:55 2018

@author: zw4215, zw4315
"""
# Q1 Eigen faces
# a)Implementing convebtional PCA, geting the best M eigen faces

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


M = 20
eivec = eivec.transpose()          # transpose it to sort the eigen vectors
indexEigenvalue = np.argsort(eival)
pcaEigenval = eival[indexEigenvalue[-M:]]
pcaEigenvec = eivec[indexEigenvalue[-M:]]
bestpcaEigenvec = pcaEigenvec[::-1]     # print the eigenface with high energy first

for i in range(0, 20):
    pic = np.swapaxes( np.reshape( np.array(bestpcaEigenvec[i,:]), (46, 56) ), 0, 1)
    plt.subplot(4,5,i+1)
    plt.axis('off')
    plt.imshow(abs(pic), cmap='gray')

