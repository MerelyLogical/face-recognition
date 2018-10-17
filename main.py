# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:41:29 2018

@author: zw4{2,3}15
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

mat = sio.loadmat('face.mat')
data = mat.get("X", "Data read error")
label = mat.get("l", "Label read error")

data_label = np.vstack([data, label])
data_train, data_test = train_test_split( data_label.transpose(), test_size=0.2, random_state=42)  #random_state split the data in a certain way
data_train = data_train.transpose()
data_test = data_test.transpose()

for i in range(0, 50):
    pic = np.swapaxes(np.reshape(np.array(data[:,i]), (46, 56)), 0, 1)
    plt.subplot(5,10,i+1)
    plt.axis('off')
    plt.imshow(pic, cmap='gray')
