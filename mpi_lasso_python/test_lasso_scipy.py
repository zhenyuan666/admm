from __future__ import division
import numpy as np
from numpy.linalg import norm
import math
from sklearn import linear_model


"""
"""
main_folder = '/global/cscratch1/sd/zhenyuan/mpi_lasso_python/data1/'
main_folder = '/Users/zhenyuanliu/Dropbox/fall2017/EE227BT/project/mpi_lasso_python/data1/'
i = 0

filename_A = main_folder + 'A' + str(i) + '.npy'
A = np.load(filename_A)
filename_b = main_folder + 'b' + str(i) + '.npy'
b = np.load(filename_b)
#
m, n = A.shape
mylambda = 0.5
clf = linear_model.Lasso(alpha = mylambda/m)
clf.fit(A, b)
x_hat = clf.coef_.reshape((clf.coef_.shape[0], 1))
# calculate the objective function value
obj = 1./2 * norm(np.dot(A, x_hat) - b)**2 + mylambda * norm(x_hat, 1)
print('objective value is : ' + str(obj))
print('norm of x_hat is: ' + str(norm(x_hat)))








