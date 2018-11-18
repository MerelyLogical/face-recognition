#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:44:56 2018
# probability of assignment to each class
@author: edward
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

data_train = np.load("data_train.npy");
label_train = np.load("label_train.npy");
data_test = np.load("data_test.npy");
label_test = np.load("label_test.npy");



#Computing the eigenvalue and eigenvector of the training data
mean_face = np.mean(data_train, axis=1)        
phi_matrix = data_train[:,0] - mean_face   

for i in range (1,data_train.shape[1]):
   column_sub = data_train[:,i] - mean_face
   phi_matrix = np.column_stack((phi_matrix, column_sub))

LDS = (np.dot(phi_matrix.transpose(), phi_matrix))/phi_matrix.shape[1]      #(1/N)*ATA
LDeival, LDeivec = np.linalg.eig(LDS)          #low -dimensional computation

Positive_eigenval = 0
for i in range(len(LDeival)):
    if LDeival[i] > 0.00001:
        Positive_eigenval = Positive_eigenval + 1          #Positive_eigenval = N-1
#--------------------------------------------------------------------------
#generate T random subspace
# the first M0 dimensions are fixed as the M0 largest eigenspace in W
M0 = 45
M1 = 105
T_amount = 10
random_subspace = []
for T in range(0,T_amount):
    LDeivec = LDeivec.transpose()
    LDindexEigenvalue = np.argsort(LDeival)
    
    LDpcaEigenval = LDeival[LDindexEigenvalue[-M0:]]
    LDpcaEigenvec = LDeivec[LDindexEigenvalue[-M0:]]
    
    #the remaining M1 dimension are randomly selected from the other N-1-M0 eigen faces in W
    # M1 can only be chosen from a certain range
    #M1 range + M0 = PV
    M1_Range = LDindexEigenvalue[(len(LDindexEigenvalue)-Positive_eigenval):(len(LDindexEigenvalue)-M0)]
    M1_list = []
    for i in range(len(M1_Range)):
        M1_list.append(M1_Range[i])
    randomlist = random.sample(M1_list, M1)
    
    for i in randomlist:
        LDpcaEigenval = np.append(LDpcaEigenval,LDeival[i])
        LDpcaEigenvec = np.vstack((LDpcaEigenvec,LDeivec[i]))
    random_subspace.append(LDpcaEigenvec)
#------------------------------------------------------------------------------
# From the T random spaces, constructing T LDA classifiers
# reconstruct D-dimension vector first,A*v = u
Bestpca_T =[]
Mpca = M0+M1

for T in range(0,T_amount):
    Bestpca_T .append([])
    LDreeigenvec = np.zeros(2576)
    for i in range(0, Mpca):
        LDvecdir = np.dot(phi_matrix, random_subspace[T][i,:])      # A*v = u
        LDnormed = LDvecdir/np.linalg.norm(LDvecdir) 
        LDreeigenvec = np.column_stack((LDreeigenvec ,LDnormed))
    bestLDpcaEigenvec = LDreeigenvec.transpose()[::-1][:-1,:]  
    Bestpca_T[T].append(bestLDpcaEigenvec)
#----------------------------------------------------------------------------------
#perform lda in each sub space
#find class mean in D-dimensional space
fish_array = []

for T in range(0,T_amount):
    bestpcaEigenvec = Bestpca_T[T][0]
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
    eivec_lda = eivec_lda.transpose()          # transpose it to sort the eigen vectors
    indexEigenvalue_lda = np.argsort(eival_lda)
  
    for M_lda in range (30,31):
        ldaEigenval = eival_lda[indexEigenvalue_lda[-M_lda:]]
        ldaEigenvec = eivec_lda[indexEigenvalue_lda[-M_lda:]]
        bestldaEigenvec = ldaEigenvec[::-1]     # print the eigenface with high energy first
        bestldaEigenval = abs(ldaEigenval[::-1]) 
        
        # force printing by recovering pca
        fish = np.dot(bestldaEigenvec, bestpcaEigenvec)
    fish_array.append(fish) 
         
#----------------------------------------------------------------------------------------
# caculate the probability x assigned to each class
P_array = []
for T in range(0,T_amount):
    Wki = np.zeros(30)
    Wkt = fish_array[T]
    for class_index in range(0,52):
        mi = m_arr[:,class_index]
        single_Wki = np.dot( Wkt, mi)
        Wki = np.column_stack((Wki,single_Wki))
    Wki = np.delete(Wki, 0, 1)
        
    
    for test_img in range(len(label_test)):
        Wkx = np.dot(Wkt, data_test)
    
    P = 0.5*(1+np.dot(Wkx.transpose(),Wki)/((np.linalg.norm(Wkx, ord=2))*(np.linalg.norm(Wki, ord=2))))
    P_array.append(P)
  
    
#----------------------------------------------------------------------------------------
# Fusion rule: sum
et_array = []
P_total = np.zeros(shape = (104,52))
for Pro in range(0,T_amount):
    single_class_pro = np.argmax(P_array[Pro], axis=1) + 1
    single_pred = (np.count_nonzero(single_class_pro - label_test))
    P_total = P_array[Pro] + P_total

P_total = P_total/T_amount
class_pro = np.argmax(P_total, axis=1) + 1

fails = (np.count_nonzero(class_pro - label_test))
cm_successrate = 1 - (fails / len(label_test))
print('Sum fusion: '+str(cm_successrate))


#----------------------------------------------------------------------------------------
