#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:06:15 2018

@author: edward
"""

#Q1 Application of Eigen faces 
#b) NN  
import numpy as np
import matplotlib.pyplot as plt

#load the variables from files
data_train = np.load("data_train.npy");
label_train = np.load("label_train.npy");
data_test = np.load("data_test.npy");
label_test = np.load("label_test.npy");
phi_matrix = np.load("phi_matrix.npy");
LDeival = np.load("LDeival.npy");
LDeivec = np.load("LDeivec.npy");

M = 40

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

mean_face = np.mean(data_train, axis=1) 

W_train= np.zeros(M)                            #Build overall W for the whole training set, 416 pics, W should be M rows, 416 columns


for pic_idx in range (0, len(label_train)):
    LDai = np.dot(phi_matrix[:,pic_idx].transpose(), bestLDpcaEigenvec.transpose()[:, 0])
    single_w = LDai 
    
    for i in range(1,M):    
        LDai = np.dot(phi_matrix[:,pic_idx].transpose(), bestLDpcaEigenvec.transpose()[:, i])
        single_w = np.column_stack((single_w,LDai))           #this is the w for each pic using best M eigen vectors
        
    W_train = np.column_stack((W_train, single_w.transpose()))    #Overall W for the whole training set, 416 pics, W should be M rows, 416 columns
  

loss_fun = 0                   # the following is projecting each pic in test set onto eigen vector
for test_idx in range(0, len(label_test)):
    test_pic = data_test[:,test_idx]    
    nrom_test_pic = test_pic - mean_face    #normalize x in test set

    test_ai= np.dot(nrom_test_pic.transpose(), bestLDpcaEigenvec.transpose()[:, 0])  #ai is a scalar
    test_w = test_ai                   

    for i in range(1,M):    
        test_ai = np.dot(nrom_test_pic.transpose(), bestLDpcaEigenvec.transpose()[:, i])
        test_w = np.column_stack((test_w,test_ai))     # scalar ai in a row



    test_w = test_w.transpose()     #form test_w in column form

    test_arr = np.repeat(test_w , len(label_train))
    test_arr = np.swapaxes( np.reshape( np.array(test_arr), (M,len(label_train))), 0, 1)
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

