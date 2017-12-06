from __future__ import division
import numpy as np
from numpy.linalg import norm
import math
from sklearn import linear_model
from sklearn import svm

"""
"""
np.random.seed(0)
main_folder = '/global/cscratch1/sd/zhenyuan/mpi_svm_python/data1/'
main_folder = '/global/cscratch1/sd/zhenyuan/mpi_svm_python/data24/'
main_folder = '/Users/zhenyuanliu/Dropbox/fall2017/EE227BT/project/mpi_svm_python/data4/'
main_folder = '/Users/zhenyuanliu/Dropbox/fall2017/EE227BT/project/mpi_svm_python/data1/'
m = 12800   # number of rows
n = 100    # number of columns
num_process = 1 # number of processes
num_per_batch = int(m/num_process)
#
X = np.random.normal(0, 1, (int(1.2 * m), n))
w_true = np.random.normal(0, 1, (n, 1))
b_true = 3.
index_accept = np.where(abs(np.dot(X, w_true) + b_true) > 1e-1)[0]
m2 = index_accept.shape[0]
temp2 = np.random.choice(m2, m, replace = False)
index_use = index_accept[temp2]
#
X = X[index_use, :]
#
y = np.zeros((m, 1))
index_positive = np.where(np.dot(X, w_true) + b_true > 0)[0]
index_negative = np.where(np.dot(X, w_true) + b_true < 0)[0]
y[index_positive, :] = 1
y[index_negative, :] = -1
#y = y.ravel()
print(index_positive.shape)

for i in range(num_process):
    filename_X = main_folder + 'X' + str(i)
    np.save(filename_X, X[i * num_per_batch: (i + 1) * num_per_batch, :])
    filename_y = main_folder + 'y' + str(i)
    np.save(filename_y, y[i * num_per_batch: (i + 1) * num_per_batch, :])












