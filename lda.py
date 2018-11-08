#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday 8th Nov 20:40:54 2018
@author: zw4215, zw4315
"""
# Q3 LDA Ensemble for Face Recognition 
# PCA-LDA
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#load the variables from files
data_train = np.load("data_train.npy");
label_train = np.load("label_train.npy");
data_test = np.load("data_test.npy");
label_test = np.load("label_test.npy");
phi_matrix = np.load("phi_matrix.npy");
eival = np.load("eival.npy");
eivec = np.load("eivec.npy");

#find class mean in D-dimensional space
distinct_class = np.unique(label_train)                           # types of distinct faces
count_diffclass =np.zeros(len(distinct_class))                      # count amount of faces for each class
sum_mi = np.zeros(shape=(2576,1))                      # sum of faces for the same class
m = np.empty(shape=(2576,1))                              # array of class mean in D dimensional space
count = 0

for element in range(len(distinct_class)):
    for i in range (len(label_train)):
        if distinct_class[element] == label_train[i]:
           sum_mi = data_train[:,i] +sum_mi[element]
           count = count + 1
    count_diffclass[element] = count
    mi = sum_mi / count                                       #single class mean in D dimensional space
    m = np.column_stack((m, (mi)))                              #stack mi into an array m
    count = 0

for i in range(0, 20):
    pic = np.swapaxes( np.reshape( np.array(m[:,i+1]), (46, 56) ), 0, 1)
    plt.subplot(4,5,i+1)
    plt.axis('off')
    plt.imshow(abs(pic), cmap='gray')
