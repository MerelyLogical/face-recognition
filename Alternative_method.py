#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 23:02:53 2018

@author: zw4215, zw4315

"""
#Q1 Application of Eigen faces 
#b) Alternative methods using mimimum reconstruction error J


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


M = 416

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


#construct the phi matrix for the test set
mean_face = np.mean(data_train, axis=1) 
phi_test = data_test[:,0] - mean_face   

for i in range (1,data_test.shape[1]):
   column_sub = data_test[:,i] - mean_face
   phi_test = np.column_stack((phi_test, column_sub))
   
# reconstruct the image in test set  
reconstruct_test = np.zeros(2576)
for pic_idx in range (0, data_test.shape[1]):
    LDai = np.dot(phi_test[:,pic_idx].transpose(), bestLDpcaEigenvec.transpose()[:, 0])
    LDsum = mean_face +  np.dot(LDai,bestLDpcaEigenvec.transpose()[:, 0])
    
    for i in range(1,M):    
        LDai = np.dot(phi_test[:,pic_idx].transpose(), bestLDpcaEigenvec.transpose()[:, i])
        LDsum = LDsum + np.dot(LDai,bestLDpcaEigenvec.transpose()[:, i])
    
    reconstruct_test= np.column_stack((reconstruct_test, LDsum))
    reconstruct_test_final= np.delete(reconstruct_test, 0, 1)

# finding the minimum reconstruct error

for i in range(0, 20):
    pic = np.swapaxes( np.reshape( np.array( reconstruct_test_final[:,i]), (46, 56) ), 0, 1)
    plt.subplot(4,5,i+1)
    plt.axis('off')
    plt.imshow(abs(pic), cmap='gray')


    
#build test_arr for each column in the testing set,expand it to the same size as training set
loss_fun = 0   
for test_idx in range(0, len(label_test)):
    test_arr = np.repeat(reconstruct_test_final[:,test_idx], len(label_train))
    test_arr = np.swapaxes( np.reshape( np.array(test_arr), (data_test.shape[0],len(label_train))), 0, 1)
    test_arr = test_arr.transpose()    
    
    J = data_train - test_arr                #This is the reconstruction error
    
    least_error = np.linalg.norm(J[:,0])     # start from the first column


    count = 0
    for i in range(1, len(label_train)):              #finding where the mim error is, count is its index
        error = np.linalg.norm(J[:,i])
        if error < least_error:
            least_error = error
            count =i

    if label_train[count]!= label_test[test_idx]:
        loss_fun = loss_fun + 1

print(loss_fun)                
print((len(label_test) - loss_fun)/ len(label_test))     
