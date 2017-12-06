from __future__ import division
import numpy as np
from numpy.linalg import norm
import math
from sklearn import linear_model
from sklearn import svm

"""
"""
main_folder = '/global/cscratch1/sd/zhenyuan/mpi_svm_python/data1/'
main_folder = '/Users/zhenyuanliu/Dropbox/fall2017/EE227BT/project/mpi_svm_python/data1/'
i = 0

filename_X = main_folder + 'X' + str(i) + '.npy'
X = np.load(filename_X)
filename_y = main_folder + 'y' + str(i) + '.npy'
y = np.load(filename_y)

# 
m, n = X.shape
clf = svm.SVC(kernel = 'linear', verbose = True)
y_raveld = y.ravel()
clf.fit(X, y_raveld)

# calculate the objective function value
w_hat = clf.coef_
b_hat = clf.intercept_

# check that w_hat is aligned with w_true
# print(np.dot(w_hat, w_true)/norm(w_hat)/norm(w_true))

A = - np.dot(np.diag(y_raveld), np.concatenate((X, np.ones((m, 1))), axis = 1))
mA, nA = A.shape
w_hat = w_hat.T.ravel()
w_hat_intercept = np.concatenate((w_hat, np.array((b_hat))), axis = 0)
obj = 0.5 * np.linalg.norm(w_hat)**2 + np.sum(np.maximum(np.dot(A, w_hat_intercept.reshape((nA, 1))) + 1, np.zeros((mA, 1))))
print('objective value is : ' + str(obj))
print('norm of w_hat is: ' + str(norm(w_hat)))
print('norm of b_hat is: ' + str(norm(b_hat)))



