from __future__ import division
import numpy as np
from numpy.linalg import norm
import math
import scipy.io as sio

"""
"""
main_folder = '/global/cscratch1/sd/zhenyuan/mpi_lasso_python/data1/'
main_folder = '/global/cscratch1/sd/zhenyuan/mpi_lasso_python/data24/'
main_folder = '/Users/zhenyuanliu/Dropbox/fall2017/EE227BT/project/mpi_lasso_python/data1/'
main_folder = '/Users/zhenyuanliu/Dropbox/fall2017/EE227BT/project/mpi_lasso_python/data4/'

np.random.seed(0)

m = 1280    # number of rows 128000
n = 100      # number of columns 10000
num_process = 4 # number of processes
num_per_batch = int(m/num_process)

A = np.random.normal(0, 1, (m, n))
for j in range(n):
	A[:,j] = A[:, j]/norm(A[:, j])
x_true = np.random.normal(0, 1, (n, 1))
index_zero = np.random.choice(n, int(n * 0.9))
x_true[index_zero, :] = 0
v = np.random.normal(0, 1e-3, (m, 1));
b = np.dot(A, x_true) + v;


for i in range(num_process):
    filename_A = main_folder + 'A' + str(i)
    np.save(filename_A, A[i * num_per_batch: (i + 1) * num_per_batch, :])
    filename_b = main_folder + 'b' + str(i)
    np.save(filename_b, b[i * num_per_batch: (i + 1) * num_per_batch, :])










