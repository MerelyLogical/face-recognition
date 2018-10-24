#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:40:54 2018

@author: zw4215, zw4315
"""
# Q1 Eigen faces
# a) Importing data, partition the face data into training and testing data sets
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



#Computing the eigenvalue and eigenvector of the training data, storing them in files
mean_face = np.mean(data_train, axis=1)        
phi_matrix = data_train[:,0] - mean_face   

for i in range (1,data_train.shape[1]):
   column_sub = data_train[:,i] - mean_face
   phi_matrix = np.column_stack((phi_matrix, column_sub))

S = (np.dot(phi_matrix, phi_matrix.transpose()))/phi_matrix.shape[1]
eival, eivec = np.linalg.eig(S)

LDS = (np.dot(phi_matrix.transpose(), phi_matrix))/phi_matrix.shape[1]      #(1/N)*ATA
LDeival, LDeivec = np.linalg.eig(LDS)          #low -dimensional computation


#store variable into a file
np.save("data_train.npy", data_train);
np.save("label_train.npy", label_train);
np.save("data_test.npy", data_test);
np.save("label_test.npy", label_test);
np.save("phi_matrix.npy", phi_matrix);
np.save("eival.npy", eival);
np.save("eivec.npy", eivec);
np.save("LDeival.npy", LDeival);
np.save("LDeivec.npy", LDeivec);

# plots the mean face
mean_pic = np.swapaxes( np.reshape( np.array(mean_face), (46, 56) ), 0, 1)
plt.axis('off')
plt.imshow(mean_pic, cmap='gray') 
