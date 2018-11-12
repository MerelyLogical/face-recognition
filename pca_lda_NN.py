#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:10:20 2018

@author: edward
"""

#Q3  
# PCA + LDA + NN  
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
sum_mi = np.zeros(shape=(2576,1))                      # sum of faces for the same class
Si = np.zeros(shape=(2576,2576)) 
Sw = np.zeros(shape=(2576,2576)) 
Sb = np.zeros(shape=(2576,2576)) 
m_arr = np.empty(shape=(2576,1))                              # array of class mean in D dimensional space
count = 0
m = np.mean(data_train, axis=1) 

for element in range(len(distinct_class)):
    for i in range (len(label_train)):
        if distinct_class[element] == label_train[i]:
           sum_mi = data_train[:,i] +sum_mi[element]
           count = count + 1
    count_diffclass[element] = count
    mi = sum_mi / count                                       #single class mean in D dimensional space
    Sb = np.outer((mi-m),(mi-m)) + Sb
    m_arr = np.column_stack((m_arr, mi))                              #stack mi into an array m
    count = 0
m_arr = np.delete(m_arr,0,1)
class_vs_amount =  np.column_stack((distinct_class, count_diffclass))  

Sw_rank = []
for element in range(len(distinct_class)):
    for i in range (len(label_train)): 
        if distinct_class[element] == label_train[i]:
            Si = np.outer((data_train[:,i] - m_arr[:,element]),(data_train[:,i] - m_arr[:,element])) + Si
    Sw = Sw + Si
    Sw_rank.append(np.linalg.matrix_rank(Sw))
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


#----------------------------------------------------------------------------------------
mean_face = np.mean(data_train, axis=1) 
#Build overall W for the whole training set, 416 pics, W should be M rows, 416 columns
W_train= np.zeros(M_lda)                           


for pic_idx in range (0, len(label_train)):
    fishai = np.dot(phi_matrix[:,pic_idx].transpose(), fish.transpose()[:, 0])
    single_w = fishai
    
    for i in range(1,M_lda):    
        fishai = np.dot(phi_matrix[:,pic_idx].transpose(), fish.transpose()[:, i])
        single_w = np.column_stack((single_w,fishai))           #this is the w for each pic using best M eigen vectors
        
    W_train = np.column_stack((W_train, single_w.transpose()))    #Overall W for the whole training set, 416 pics, W should be M rows, 416 columns
  

loss_fun = 0                   # the following is projecting each pic in test set onto eigen vector
for test_idx in range(0, len(label_test)):
    test_pic = data_test[:,test_idx]    
    nrom_test_pic = test_pic - mean_face    #normalize x in test set #mean face

    test_ai= np.dot(nrom_test_pic.transpose(), fish.transpose()[:, 0])  #ai is a scalar
    test_w = test_ai                   

    for i in range(1,M_lda):    
        test_ai = np.dot(nrom_test_pic.transpose(), fish.transpose()[:, i])
        test_w = np.column_stack((test_w,test_ai))     # scalar ai in a row

    test_w = test_w.transpose()     #form test_w in column form

    test_arr = np.repeat(test_w , len(label_train))
    test_arr = np.swapaxes( np.reshape( np.array(test_arr), (M_lda,len(label_train))), 0, 1)
    test_arr = test_arr.transpose()    
    
    W_train_fianl = np.delete(W_train, 0, 1)   # delete the initial 0 colomn
    
    e = W_train_fianl - test_arr
    least_error = np.linalg.norm(e[:,0])     # start from the first column


    count = 0
    for i in range(1, len(label_train)):
        error = np.linalg.norm(e[:,i])
        if error < least_error:
            least_error = error
            count =i

    if label_train[count]!= label_test[test_idx]:
        loss_fun = loss_fun + 1
        

print(loss_fun)
print((len(label_test) - loss_fun)/ len(label_test))
