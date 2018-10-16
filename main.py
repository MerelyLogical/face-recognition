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

for i in range(0, 20):
    pic = np.swapaxes(np.reshape(np.array(data[:,i]), (46, 56)), 0, 1)
    plt.subplot(4,5,i+1)
    plt.axis('off')
    plt.imshow(pic, cmap='gray')

plt.show()