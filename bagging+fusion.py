#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:10:20 2018

@author: edward
"""

# Q3 randomnization 
# a) bagging 
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random

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

MDDD = np.zeros((len(label_train), len(label_test)))
#---------------bagging 
data_train_matrix = []
T_amount = 150
for T in range(0,T_amount):
    print('bagging')
    print(T)
    data_train_matrix.append([])
    class_matrix =[]       
    for i in range(1,53):
        class_matrix.append([])
        for j in range(len(label_train)):
            if i == label_train[j]:
                class_matrix[i-1].append(data_train[:,j])
    
    random_training_data = np.array(class_matrix[0][1])
    random_training_label = []
    
    for i in range(0,52):
        for j in range(0,8):
            random_training_label.append(i+1)
            index = random.randint(0,7),
            random_training_data = np.column_stack((random_training_data, class_matrix[i][index[0]])) 
    
    random_training_data = np.delete(random_training_data, 0, 1)
    data_train_matrix[T].append(random_training_data)
#-----------------------------------------------------------------------------------
#store variable into a file
np.save("random_training_data.npy", random_training_data);
np.save("random_training_label.npy", random_training_label);
np.save("data_test.npy", data_test);
np.save("label_test.npy", label_test);
#------------------------------------------------------------------------------------
#perform pca-lda with T models of random data, then use majority voting
#First computing the eigenvalue and eigenvector of the training data
label_train = random_training_label
Mpca = 150
Bestpca_T =[]
for T in range(0,T_amount):
    print('pcalda')
    print(T)
    mean_face = np.mean(data_train_matrix[T][0], axis=1)        
    phi_matrix = data_train_matrix[T][0][:,0] - mean_face   
    
    for i in range (1,data_train_matrix[T][0].shape[1]):
       column_sub = data_train_matrix[T][0][:,i] - mean_face
       phi_matrix = np.column_stack((phi_matrix, column_sub))
    
    LDS = (np.dot(phi_matrix.transpose(), phi_matrix))/phi_matrix.shape[1]      #(1/N)*ATA
    LDeival, LDeivec = np.linalg.eig(LDS)          #low -dimensional computation
    LDeivec = LDeivec.transpose()
    LDindexEigenvalue = np.argsort(LDeival)
    
    LDpcaEigenval = LDeival[LDindexEigenvalue[-Mpca:]]
    LDpcaEigenvec = LDeivec[LDindexEigenvalue[-Mpca:]]
#---------------------------------------------------------------------------------------
    Positive_eigenval = 0
    for i in range(len(LDeival)):
        if LDeival[i] > 0.00001:
            Positive_eigenval = Positive_eigenval + 1          #Positive_eigenval = N-1
    
    if(Positive_eigenval < Mpca):
        print("Mpca is too big")
#-------------------------------------------------------------------------------------

    Bestpca_T.append([])
    LDreeigenvec = np.zeros(2576)
    for i in range(0, Mpca):
        LDvecdir = np.dot(phi_matrix, LDpcaEigenvec[i,:])      # A*v = u
        LDnormed = LDvecdir/np.linalg.norm(LDvecdir) 
        LDreeigenvec = np.column_stack((LDreeigenvec ,LDnormed))
    bestLDpcaEigenvec = LDreeigenvec.transpose()[::-1][:-1,:]  
    Bestpca_T[T].append(bestLDpcaEigenvec)
#---------------------------------------------------------------------------------
#perform lda in each sub space
#find class mean in D-dimensional space
fish_array = []

for T in range(0,T_amount):
    print('fisher')
    print(T)
    data_train = data_train_matrix[T][0]
    bestpcaEigenvec = Bestpca_T[T][0]
    distinct_class = np.unique(label_train)                           # types of distinct faces
    count_diffclass =np.zeros(len(distinct_class))                      # count amount of faces for each class
    sum_mi = np.zeros(2576)                      # sum of faces for the same class
    Si = np.zeros(shape=(2576,2576)) 
    Sw = np.zeros(shape=(2576,2576)) 
    Sb = np.zeros(shape=(2576,2576)) 
    m_arr = np.empty(shape=(2576,1))                              # array of class mean in D dimensional space
    count = 0
    m = np.mean(data_train, axis=1) qqqqqqqqqqqq
    
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
    eivec_lda = eivec_lda.transpose()          # transpose it to sort the eigen vectors
    indexEigenvalue_lda = np.argsort(eival_lda)
  
    M_lda = 30
    ldaEigenval = eival_lda[indexEigenvalue_lda[-M_lda:]]
    ldaEigenvec = eivec_lda[indexEigenvalue_lda[-M_lda:]]
    bestldaEigenvec = ldaEigenvec[::-1]     # print the eigenface with high energy first
    bestldaEigenval = abs(ldaEigenval[::-1]) 
    
    # force printing by recovering pca
    fish = np.dot(bestldaEigenvec, bestpcaEigenvec)
    fish_array.append(fish) 
         
#----------------------------------------------------------------------------------------
#NN classification
success_rate_array = [] 
majority_count = []
et_array = []
#Build overall W for the whole training set, 416 pics, W should be M rows, 416 columns
for T in range(0,T_amount):
    print(T)
    data_train = data_train_matrix[T][0]
    mean_face = np.mean(data_train, axis=1)
    majority_count.append([])
    W_train= np.zeros(M_lda) 
    fish =  fish_array[T]                          
    for pic_idx in range (0, len(label_train)):
        fishai = np.dot(phi_matrix[:,pic_idx].transpose(), fish.transpose()[:, 0])
        single_w = fishai
        
        for i in range(1,M_lda):    
            fishai = np.dot(phi_matrix[:,pic_idx].transpose(), fish.transpose()[:, i])
            single_w = np.column_stack((single_w,fishai))           #this is the w for each pic using best M eigen vectors
            
        W_train = np.column_stack((W_train, single_w.transpose()))    #Overall W for the whole training set, 416 pics, W should be M rows, 416 columns
      
    
    loss_fun = 0                   # the following is projecting each pic in test set onto eigen vector
    MDD = np.zeros(len(label_train))
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
 # Doing test       
        e = W_train_fianl - test_arr
        least_error = np.linalg.norm(e[:,0])     # start from the first column
        MD = least_error
    
        count = 0
        for i in range(1, len(label_train)):
            error = np.linalg.norm(e[:,i])
            MD = np.append(MD, error)
            if error < least_error:
                least_error = error
                count = i
        
        majority_count[T].append(count)                   #record prediction for each model
        if label_train[count]!= label_test[test_idx]:
            loss_fun = loss_fun + 1     

        MDD = np.column_stack((MDD, MD))
    success_rate_array.append((len(label_test) - loss_fun)/ len(label_test))
    et_array.append(loss_fun/len(label_test))        #assume et(x)= 1 for misclassification
    MDDD = np.dstack((MDDD, MDD[:,1:]))
Eav = np.mean(et_array)
#---------------------------------------------------------------------------------------------
#Majority voting and committe 
prediction = []                                           #record prediction for each model
for image in range(len(label_test)): 
    for T in range(0,T_amount): 
        prediction.append(majority_count[T][image])

prediction = np.reshape(prediction,(104,T_amount))
com_et = 0
com_et_arr = []
for image in range(len(label_test)):
    for T in range(T_amount):
        if label_train[prediction[image,T]]!= label_test[image]:
            et = 1
        else:
            et =0
        com_et = com_et + et
    com_et = (com_et/T_amount)**2
    com_et_arr.append(com_et)
    com_et = 0
Ecom = np.mean(com_et_arr)
        
fusion = []                        # this records the result of majority voting 
l = []
for image in range(len(label_test)): 
    l = prediction[image,:]
    l = l.tolist()
    fusion.append(max(set(l), key = l.count))

mloss_fun = 0
for i in range(len(label_test)):
    if label_train[fusion[i]]!= label_test[i]:
        mloss_fun = mloss_fun + 1
    
fusion_success_rate  = (len(label_test) - mloss_fun)/ len(label_test)   
print('Voting: '+str(fusion_success_rate))
print(Eav/Ecom)
#---------------------------------------------------------------------------------------------
# Committee Machine
MDDD = MDDD[:,:,1:]
cm_results = np.zeros(len(label_test))
for test in range(0,len(label_test)):
    cm_pertestresults = np.zeros(52)
    for klas in range(0,52):
        for pik in range(0,8):
            idks = (klas * 8) + pik
            cm_pertestresults[klas] += 1 / (np.sum(MDDD[idks,test,:]))
    cm_results[test] = np.argmax(cm_pertestresults) + 1
    
fails = (np.count_nonzero(cm_results - label_test))
cm_successrate = 1 - (fails / len(label_test))
print('Committee: '+str(cm_successrate))