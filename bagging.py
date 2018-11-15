
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

#bagging 
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
        random_training_label.append(i)
        index = random.randint(0,7),
        random_training_data = np.column_stack((random_training_data, class_matrix[i][index[0]])) 

    


#store variable into a file
np.save("random_training_data.npy", random_training_data);
np.save("random_training_label.npy", random_training_label);
np.save("data_test.npy", data_test);
np.save("label_test.npy", label_test);


