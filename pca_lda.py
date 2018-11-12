#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday 8th Nov 20:40:54 2018
@author: zw4215, zw4315
"""
# Q3 LDA Ensemble for Face Recognition 
# PCA-LDA
import numpy as np
import matplotlib.pyplot as plt

#load the variables from files
data_train = np.load("data_train2.npy");
label_train = np.load("label_train2.npy");
data_test = np.load("data_test2.npy");
label_test = np.load("label_test2.npy");
phi_matrix = np.load("phi_matrix2.npy");
eival = np.load("eival2.npy");
eivec = np.load("eivec2.npy");

#--------------------------------------------------------------------------------
# Doing PCA first 
M_pca = 25
eivec = eivec.transpose()          # transpose it to sort the eigen vectors
indexEigenvalue = np.argsort(eival)
pcaEigenval = eival[indexEigenvalue[-M_pca:]]
pcaEigenvec = eivec[indexEigenvalue[-M_pca:]]
bestpcaEigenvec = pcaEigenvec[::-1]     # print the eigenface with high energy first
bestpcaEigenval = abs(pcaEigenval[::-1]) 

#--------------------------------------------------------------------------------

#find class mean in D-dimensional space
distinct_class = np.unique(label_train)                           # types of distinct faces
count_diffclass =np.zeros(len(distinct_class))                      # count amount of faces for each class
sum_mi = np.zeros(2576)                      # sum of faces for the same class
Si = np.zeros(shape=(2576,2576)) 
Sw = np.zeros(shape=(2576,2576)) 
Sb = np.zeros(shape=(2576,2576)) 
m_arr = np.empty(shape=(2576,1))                              # array of class mean in D dimensional space
count = 0
m = np.mean(data_train, axis=1) 

for element in range(len(distinct_class)):
    for i in range (len(label_train)):
        if distinct_class[element] == label_train[i]:
           sum_mi = data_train[:,i] + sum_mi
           count = count + 1
    count_diffclass[element] = count
    mi = sum_mi / count                                       #single class mean in D dimensional space
    sum_mi = np.zeros(2576) 
    Sb = np.outer((mi-m),(mi-m)) + Sb
    m_arr = np.column_stack((m_arr, mi))                              #stack mi into an array m
    count = 0
m_arr = np.delete(m_arr,0,1)
class_vs_amount =  np.column_stack((distinct_class, count_diffclass))  

#Sw_rank = []
for element in range(len(distinct_class)):
    for i in range (len(label_train)): 
        if distinct_class[element] == label_train[i]:
            Si = np.outer((data_train[:,i] - m_arr[:,element]),(data_train[:,i] - m_arr[:,element])) + Si
    Sw = Sw + Si
    #Sw_rank.append(np.linalg.matrix_rank(Sw))
    Si = np.zeros(shape=(2576,2576))        


Sw_pca = np.dot(np.dot(bestpcaEigenvec,Sw),bestpcaEigenvec.transpose())
Sb_pca = np.dot(np.dot(bestpcaEigenvec,Sb),bestpcaEigenvec.transpose())

S_lda = np.dot(np.linalg.inv(Sw_pca),Sb_pca) 
eival_lda, eivec_lda = np.linalg.eig(S_lda)

M_lda = 20
eivec_lda = eivec_lda.transpose()          # transpose it to sort the eigen vectors
indexEigenvalue_lda = np.argsort(eival_lda)
ldaEigenval = eival_lda[indexEigenvalue_lda[-M_lda:]]
ldaEigenvec = eivec_lda[indexEigenvalue_lda[-M_lda:]]
bestldaEigenvec = ldaEigenvec[::-1]     # print the eigenface with high energy first
bestldaEigenval = abs(ldaEigenval[::-1]) 

# force printing by recovering pca
fish = np.dot(bestldaEigenvec, bestpcaEigenvec)

for i in range(0, 20):
    pic = np.swapaxes( np.reshape( np.array(fish[i,:]), (46, 56) ), 0, 1)
    plt.subplot(4,5,i+1)
    plt.axis('off')
    plt.imshow(abs(pic), cmap='gray')
 
# test
