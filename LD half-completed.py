# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:41:29 2018
@author: zw4{2,3}15
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

mat = sio.loadmat('face.mat')
data = mat.get("X", "Data read error")
label = mat.get("l", "Label read error")


data_label = np.vstack([data, label])
data_train, data_test = train_test_split( data_label.transpose(), test_size=0.2, random_state=42)  #random_state split the data in a certain way

data_train = data_train.transpose()
data_test = data_test.transpose()


label_train = data_train[2576,:]
label_test = data_test[2576,:]
data_train = np.delete(data_train, 2576, 0)
data_test = np.delete(data_test, 2576, 0)


mean_face = np.mean(data_train, axis=1) 
phi_matrix = data_train[:,0] - mean_face


for i in range (1,data_train.shape[1]):
   column_sub = data_train[:,i] - mean_face
   phi_matrix = np.column_stack((phi_matrix, column_sub))
   

S = (np.dot(phi_matrix, phi_matrix.transpose()))/phi_matrix.shape[1]
eival, eivec = np.linalg.eig(S)

LDS = (np.dot(phi_matrix.transpose(), phi_matrix))/phi_matrix.shape[1]      #(1/N)*ATA
LDeival, LDeivec = np.linalg.eig(LDS)          #low -dimensional computation

eivec = eivec.transpose()
LDeivec = LDeivec.transpose()

indexEigenvalue = np.argsort(eival)
LDindexEigenvalue = np.argsort(LDeival)

M = 20
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

bestpcaEigenval = pcaEigenval[::-1]
bestLDpcaEigenval = LDpcaEigenval[::-1]   # print the eigenface with high energy first

"""
mean_pic = np.swapaxes( np.reshape( np.array(mean_face), (46, 56) ), 0, 1)
plt.axis('off')
plt.imshow(mean_pic, cmap='gray') 
"""


#reconstruction of the image
for pic_idx in range (0, 18):
    ai = np.dot(phi_matrix[:,pic_idx].transpose(), bestpcaEigenvec.transpose()[:, 0])
    LDai = np.dot(phi_matrix[:,pic_idx].transpose(), bestLDpcaEigenvec.transpose()[:, 0])
    
    nsum = mean_face +  np.dot(ai,bestpcaEigenvec.transpose()[:, 0])
    LDsum = mean_face +  np.dot(LDai,bestLDpcaEigenvec.transpose()[:, 0])
    
    for i in range(1,M):
        ai = np.dot(phi_matrix[:,pic_idx].transpose(), bestpcaEigenvec.transpose()[:, i])
        nsum = nsum + np.dot(ai,bestpcaEigenvec.transpose()[:, i])
        
        LDai = np.dot(phi_matrix[:,pic_idx].transpose(), bestLDpcaEigenvec.transpose()[:, i])
        LDsum = LDsum + np.dot(LDai,bestLDpcaEigenvec.transpose()[:, i])
    
    
    reconstruct_pic = np.swapaxes( np.reshape( np.array(nsum), (46, 56)), 0, 1)
    original_pic =  np.swapaxes( np.reshape( np.array(data_train[:,pic_idx]), (46, 56)), 0, 1)
    LDreconstruct_pic =  np.swapaxes( np.reshape( np.array(LDsum), (46, 56)), 0, 1)
    
    plt.subplot(6,9,3*pic_idx+1)
    plt.axis('off')
    plt.imshow(abs(original_pic ),cmap='gray')
    
    plt.subplot(6,9,3*pic_idx+2)
    plt.axis('off')
    plt.imshow(abs(reconstruct_pic),cmap='gray')
    
    plt.subplot(6,9,3*pic_idx+3)
    plt.axis('off')
    plt.imshow(abs(LDreconstruct_pic),cmap='gray')
