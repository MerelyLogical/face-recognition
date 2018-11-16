#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:01:26 2018
Q1 Application of eigenfaces
b) Alternative method to NN classification
@author: zw4215, zw4315
"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

mat = sio.loadmat('face.mat')
data = mat.get("X", "Data read error")
label = mat.get("l", "Label read error")

np.random.seed(42)

data_label = np.vstack([data, label]).reshape(2577, 52, 10)
for i in range (52):
    class_slice = data_label[:, i, :].transpose()
    np.random.shuffle(class_slice)

data_train = np.concatenate(data_label[:, :, 0:8].transpose(), 0).transpose()
data_test = np.concatenate(data_label[:, :, 8:10].transpose(), 0).transpose()
label_train = data_train[2576,:]
label_test = data_test[2576,:]

data_train = np.delete(data_train, 2576, 0)
data_test = np.delete(data_test, 2576, 0)

class_matrix =[]       
for i in range(1,53):
    class_matrix.append([])
    for j in range(len(label_train)):
        if i == label_train[j]:
            class_matrix[i-1].append(data_train[:,j])

total_face = np.zeros(2576)
mean_face_matrix = np.zeros(shape=(2576,52))
for i in range(len(class_matrix)):
    for j in range(0,8):
        total_face = class_matrix[i][j] + total_face 
    mean_face = total_face /8
    mean_face_matrix[:,i] = mean_face_matrix[:,i] + mean_face
    total_face = np.zeros(2576)

succ_rate_array = []
M_arr =[]
for M in range(1,9):
    besteigenvec_matrix = [] 
    besteigenval_matrix = []  
    M_arr.append(M)
    for i in range(0,52):
        Ai = class_matrix[i][0] - mean_face_matrix[:,i]  
    
        for j in range (1,8):
            column_sub = class_matrix[i][j] - mean_face_matrix[:,i]  
            Ai = np.column_stack((Ai, column_sub))
    
        LDS = (np.dot(Ai.transpose(), Ai))/Ai.shape[1]      #(1/N)*ATA
        LDeival, LDeivec = np.linalg.eig(LDS)          #low -dimensional computation
    
       # eigenvec_matrix.append(LDeivec)
       # eigenval_matrix.append(LDeival)
    
        LDeivec = LDeivec.transpose()
        LDindexEigenvalue = np.argsort(LDeival)
        
        LDpcaEigenval = LDeival[LDindexEigenvalue[-M:]]
        LDpcaEigenvec = LDeivec[LDindexEigenvalue[-M:]]
        
        
        
        LDreeigenvec = np.zeros(2576)
        for i in range(0, M):
            LDvecdir = np.dot(Ai, LDpcaEigenvec[i,:])      # A*v = u
            LDnormed = LDvecdir/np.linalg.norm(LDvecdir) 
            LDreeigenvec = np.column_stack((LDreeigenvec ,LDnormed))
            
            
        
        bestLDpcaEigenvec = LDreeigenvec.transpose()[::-1][:-1,:]   
        bestLDpcaEigenval = LDpcaEigenval[::-1]        # print the eigenface with high energy first
        besteigenvec_matrix.append(bestLDpcaEigenvec)
        besteigenval_matrix.append(bestLDpcaEigenval)  
    
     #construct the phi matrix for the test set
    all_class_reconstruct = []
    for i in range(0,52):
        all_class_reconstruct.append([])
        mean_face = mean_face_matrix[:,i]
        phi_test = data_test[:,0] - mean_face   
        
        for j in range (1,data_test.shape[1]):
           column_sub = data_test[:,j] - mean_face
           phi_test = np.column_stack((phi_test, column_sub))
           
        # reconstruct the image in test set  
        reconstruct_test = np.zeros(2576)
        for pic_idx in range (0, data_test.shape[1]):
            LDai = np.dot(phi_test[:,pic_idx].transpose(), besteigenvec_matrix[i].transpose()[:, 0])
            LDsum = mean_face +  np.dot(LDai,besteigenvec_matrix[i].transpose()[:, 0])
            
            for k in range(1,M):    
                LDai = np.dot(phi_test[:,pic_idx].transpose(), besteigenvec_matrix[i].transpose()[:, k])
                LDsum = LDsum + np.dot(LDai,besteigenvec_matrix[i].transpose()[:, k])
            
            reconstruct_test= np.column_stack((reconstruct_test, LDsum))
            reconstruct_test_final= np.delete(reconstruct_test, 0, 1)
        all_class_reconstruct[i].append(reconstruct_test_final)
    
    
    #np.array(all_class_reconstruct[0][0][:,3]
    """
    xx = 7
    pic =  np.swapaxes( np.reshape( np.array(all_class_reconstruct[15][0][:,xx]), (46, 56) ), 0, 1)
    pic2 = np.swapaxes(np.reshape( np.array(data_test[:,xx]), (46, 56)), 0, 1)
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(pic, cmap='gray') 
    plt.subplot(1,2,2)
    plt.imshow(pic2, cmap='gray') 
    """
    
    
    error_array = []
    count = 0
    count_array = []
    
    #for each image in test set
    for i in range(0,104):
        count = 1
        least_error = np.linalg.norm(data_test[:, i] - all_class_reconstruct[0][0][:,i])
        for j in range(1,52):
            error = np.linalg.norm(data_test[:, i] - all_class_reconstruct[j][0][:,i])
            #if i == 7 and j == 7 :
                   # print(88888888, i,j,error,least_error,count)
            if error < least_error:             # found the class has minimum recon error
                least_error = error
                count = j + 1
               # if i == 7:
                 #   print(i,j,error,least_error,count)
        count_array.append(count)
    
    loss = 0
    for image in range(0,104):
        if label_test[image] != count_array[image]:
            loss = loss + 1
    
    success_rate =  (len(label_test) - loss)/len(label_test)
    succ_rate_array.append(success_rate)
    print(success_rate)
plt.plot(M_arr,succ_rate_array)
