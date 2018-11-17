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
from mpl_toolkits.mplot3d import Axes3D
import random

#load the variables from files
data_train = np.load("data_train.npy");
label_train = np.load("label_train.npy");
data_test = np.load("data_test.npy");
label_test = np.load("label_test.npy");
phi_matrix = np.load("phi_matrix.npy");
LDeival = np.load("LDeival.npy");
LDeivec = np.load("LDeivec.npy");

M = 3

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
#-------------------------------------------------------------------------------------------------
loss_fun = 0                   # the following is projecting each pic in test set onto eigen vector
f_test_arr = []
prediction_idx = []
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
        f_test_arr.append(test_idx)                 # f_test_arr is the idx which leads to misclassification
        prediction_idx.append(count)
print(loss_fun)
print((len(label_test) - loss_fun)/ len(label_test))
#-------------------------------------------------------------------------------------------------
# finding class of misclassication 
ground_truth = []
prediction = []
for i in range(len(f_test_arr)):
   ground_truth.append( label_test[f_test_arr[i]])
   prediction.append(label_train[prediction_idx[i]])
#------------------------------------------------------------------------------

iindex = 48#random.randint(0,81)

#------------------------------------------------------------------------------
#projection images in class A
right_class = np.zeros(2576)
for i in range (len(label_train)):
    if label_train[i] == ground_truth[iindex]:         #class A     ground truth
        right_class = np.column_stack((right_class,data_train[:,i]))
right_class = np.delete(right_class,0,1)

mean_face = np.mean(data_train, axis=1)

LDai_array = []
u1 = []
u2 = []
u3 = []
for pic_idx in range (0,8):
    LDai_array.append([])
    for i in range(0,M):  
        LDai = np.dot(right_class[:,pic_idx].transpose(), bestLDpcaEigenvec.transpose()[:, i])
        LDai_array[pic_idx].append(LDai)
for pic_idx in range (0,8):
    u1.append(LDai_array[pic_idx][0])
    u2.append(LDai_array[pic_idx][1])
    u3.append(LDai_array[pic_idx][2])
#------------------------------------------------------------------------------
#projection images in class B
right_class1 = np.zeros(2576)
for i in range (len(label_train)):
    if label_train[i] == prediction[iindex]:              #class B     false class
        right_class1 = np.column_stack((right_class1, data_train[:,i]))
right_class1 = np.delete(right_class1,0,1)

mean_face = np.mean(data_train, axis=1)

LDai_array1 = []
uu1 = []
uu2 = []
uu3 = []
for pic_idx in range (0,8):
    LDai_array1.append([])
    for i in range(0,M):  
        LDaai = np.dot(right_class1[:,pic_idx].transpose(), bestLDpcaEigenvec.transpose()[:, i])
        LDai_array1[pic_idx].append(LDaai)
for pic_idx in range (0,8):
    uu1.append(LDai_array1[pic_idx][0])
    uu2.append(LDai_array1[pic_idx][1])
    uu3.append(LDai_array1[pic_idx][2])
#-------------------------------------------------------------------------------

uuu1 = []
uuu2 = []
uuu3 = []
iiid = f_test_arr[iindex]                           #point be misclasified
ttest_idx = []
for i in range(0,M):  
    LDaaai = np.dot(data_test[:,iiid].transpose(), bestLDpcaEigenvec.transpose()[:, i])
    ttest_idx.append(LDaaai)
uuu1.append(ttest_idx[0])
uuu2.append(ttest_idx[1])
uuu3.append(ttest_idx[2])       
#-------------------------------------------------------------------------------
uuuu1 = []
uuuu2 = []
uuuu3 = []
iiiid = 12                   #point be correctly clasified
ttest_idx = []
for i in range(0,M):  
    LDaaaai = np.dot(data_test[:,iiiid].transpose(), bestLDpcaEigenvec.transpose()[:, i])
    ttest_idx.append(LDaaaai)
uuuu1.append(ttest_idx[0])
uuuu2.append(ttest_idx[1])
uuuu3.append(ttest_idx[2])       
#-------------------------------------------------------------------------------

fig1 = plt.figure()

ax = Axes3D(fig1)
ax.scatter3D(np.array(u1), np.array(u2), np.array(u3), c = 'red',label='Class A' )# ground truth
ax.scatter3D(np.array(uu1), np.array(uu2), np.array(uu3), marker = ',', c = 'b',label='Class B') # false class
ax.scatter3D(np.array(uuu1), np.array(uuu2), np.array(uuu3), marker = '^', c = 'y',label='Point A') #misclasified
ax.scatter3D(np.array(uuuu1), np.array(uuuu2), np.array(uuuu3), marker = 'v', c = 'black',label='Point B') #correctly clasified
#ax.plot3D(np.array(u1), np.array(u2), np.array(u3))
ax.set_title("Classification Example")
ax.legend()
ax.set_xlabel("u1")
ax.set_ylabel("u2")
ax.set_zlabel("u3")

    
