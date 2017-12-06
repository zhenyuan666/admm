from __future__ import division
import pdb,time
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
import scipy
import math
from sklearn import svm
import scipy.io as sio
from cvxpy import *

"""


"""
main_folder = '/Users/zhenyuanliu/Dropbox/fall2017/EE227BT/project/mpi_svm_python/data1/'
i = 0
filename_X = main_folder + 'X' + str(i) + '.npy'
X = np.load(filename_X)
filename_y = main_folder + 'y' + str(i) + '.npy'
y = np.load(filename_y)
#
m, n1 = X.shape
n = n1 + 1 # account for the intercept term
N = 10
num_per_batch = int(m/N)


def main():
  # now solve the problem
  z, h = svm_admm(X, y)
  print('norm of w_hat is: ' + str(np.linalg.norm(z[0:-1, 0])))
  print('norm of b_hat is: ' + str(z[-1, 0]))

"""

"""
def svm_admm(X, y, mylambda=1., rho=1., rel_par=1., QUIET = False, MAX_ITER = 200, ABSTOL = 1e-6, RELTOL = 1e-2):
    """
     linear_svm   Solve linear support vector machine (SVM) via ADMM

    [x, history] = linear_svm(A, lambda, p, rho, alpha)

    Solves the following problem via ADMM:

      minimize   (1/2)||w||_2^2 + \lambda sum h_j(w, b)

    where A is a matrix given by [-y_j*x_j -y_j], lambda is a
    regularization parameter, and p is a partition of the observations in to
    different subsystems.

    The function h_j(w, b) is a hinge loss on the variables w and b.
    It corresponds to h_j(w,b) = (Ax + 1)_+, where x = (w,b).

    This function implements a *distributed* SVM that runs its updates
    serially.

    The solution is returned in the vector x = (w,b).

    history is a structure that contains the objective value, the primal and
    dual residual np.linalg.norms, and the tolerances for the primal and dual residual
    np.linalg.norms at each iteration.

    rho is the augmented Lagrangian parameter.

    alpha is the over-relaxation parameter (typical values for alpha are
    between 1.0 and 1.8).

    More information can be found in the paper linked at:
    http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html

    """
    if not QUIET:
        tic = time.time()
    m, n = X.shape 
    y_raveld = y.ravel() 
    # A is a matrix given by [-y_j*x_j -y_j]
    A = - np.dot(np.diag(y_raveld), np.concatenate((X, np.ones((m, 1))), axis = 1))

    #Data preprocessing
    m, n = A.shape
    
    #ADMM solver
    x = np.zeros((n, N))
    z = np.zeros((n, N))
    u = np.zeros((n, N))

    if not QUIET:
        print('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' %('iter',
                                                      'r np.linalg.norm', 
                                                      'eps pri', 
                                                      's np.linalg.norm', 
                                                      'eps dual', 
                                                      'objective'))

    # Saving state
    h = {}
    h['objval']     = np.zeros(MAX_ITER)
    h['r_norm']     = np.zeros(MAX_ITER)
    h['s_norm']     = np.zeros(MAX_ITER)
    h['eps_pri']    = np.zeros(MAX_ITER)
    h['eps_dual']   = np.zeros(MAX_ITER)

    for k in range(MAX_ITER):
        # x-update 
        for i in range(N):
            A_temp = A[i * num_per_batch: (i + 1) * num_per_batch, :]
            y_temp = y[i * num_per_batch: (i + 1) * num_per_batch, :]
            #
            # temp1 = -z[:, i] + u[:, i]
            # fun = lambda x: np.sum(np.maximum(np.dot(A_temp, x.reshape((n, 1))) + 1, np.zeros((num_per_batch, 1)))) + \
            # rho/2. * np.dot(x + temp1, x  + temp1)
            # # np.random.uniform(-1, 1, (n,1))
            # result = scipy.optimize.minimize(fun, 0.1 * np.ones((n, 1)), tol = 1e-8, method = 'Nelder-Mead')
            # x_temp = result.x
            #
            x_var = Variable(n)
            constraints = []
            objective = Minimize(sum_entries(pos( A_temp * x_var + 1)) + rho/2. * sum_squares((x_var - z[:, i] + u[:, i])))
            prob = Problem(objective, constraints)
            result = prob.solve()
            x_temp = x_var.value

            x_temp = x_temp.reshape((x_temp.shape[0], 1))
            x[:, i] = x_temp.ravel()

        xave = np.mean(x, axis = 1)

        # z-update
        zold = np.copy(z)
        x_hat = rel_par * x + (1. - rel_par) * zold
        z = N * rho/(1./mylambda + N * rho) * np.mean(x_hat + u, axis = 1)
        z = z.reshape((z.shape[0], 1))
        z = np.dot(z, np.ones((1, N))) # N columns of the same values

        # u-update
        u = u + x_hat - z

        # diagnostics, reporting, termination checks
        h['objval'][k]   = myobjective(A, mylambda, x, z)
        h['r_norm'][k]   = np.linalg.norm(x - z)
        h['s_norm'][k]   = np.linalg.norm(rho * (z - zold))
        h['eps_pri'][k]  = np.sqrt(n) * ABSTOL+ RELTOL * np.maximum(np.linalg.norm(x), np.linalg.norm(-z))
        h['eps_dual'][k] = np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(rho * u)
        if not QUIET:
            print('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' %(k + 1,\
                                                          h['r_norm'][k],\
                                                          h['eps_pri'][k],\
                                                          h['s_norm'][k],\
                                                          h['eps_dual'][k],\
                                                          h['objval'][k]))

        if (h['r_norm'][k] < h['eps_pri'][k]) and (h['s_norm'][k] < h['eps_dual'][k]):
            break

    if not QUIET:
        toc = time.time()-tic
        print("\nElapsed time is %.2f seconds"%toc)

    return z, h

def myobjective(A, mylambda, x, z):
    return 0.5 * np.linalg.norm(z[0:-1, 0])**2 + mylambda * hinge_loss(A, x)

def hinge_loss(A, x):
    val = 0
    for i in range(N):
        A_temp = A[i * num_per_batch: (i + 1) * num_per_batch, :]
        val = val + np.sum(np.maximum(np.dot(A_temp, x[:, i].reshape((n, 1))) + 1, np.zeros((A_temp.shape[0], 1))))
    return val

if __name__=='__main__':
    main()