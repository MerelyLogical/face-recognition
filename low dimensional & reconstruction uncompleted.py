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


sorceEigenvalue = np.argsort(eival)
LDsorceEigenvalue = np.argsort(LDeival)


M = 20
EigenVal_difference =  sorceEigenvalue[-M:] - LDsorceEigenvalue[-M:]   # success, so low dimensional computation gets the same eigen value

pcaEigenvector = eivec[sorceEigenvalue[-M:]]
LDpcaEigenvector = LDeivec[LDsorceEigenvalue[-M:]]

## failed,should get zero :EigenVec_difference =  pcaEigenvector.transpose()[:, 0] - np.dot(phi_matrix, LDpcaEigenvector.transpose()[:, 0])

pcaEigenvector = pcaEigenvector[::-1]
LDpcaEigenvector = LDpcaEigenvector[::-1]   # print the face with high energy first


"""
mean_pic = np.swapaxes( np.reshape( np.array(mean_face), (46, 56) ), 0, 1)
plt.axis('off')
plt.imshow(mean_pic, cmap='gray') 
"""
"""
ai = np.dot(phi_matrix.transpose(), pcaEigenvector.transpose()[:, 0])
sum = mean_face +  np.dot(ai,pcaEigenvector.transpose()[:, 0])
for i in range(1,M):
    ai = np.dot(phi_matrix.transpose(), pcaEigenvector.transpose()[:, i])
    sum = sum +np.dot(ai,pcaEigenvector.transpose()[:, i])
"""    #feel something wrong with the dimension


for i in range(0, 20):
    pic = np.swapaxes( np.reshape( np.array(pcaEigenvector[i,:]), (46, 56) ), 0, 1)
    plt.subplot(4,5,i+1)
    plt.axis('off')
    plt.imshow(abs(pic),cmap='gray')
