from __future__ import division
import pdb,time,os
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky
from mpi4py import MPI
import scipy.io as sio
from cvxpy import *

"""

"""

def main():
    main_folder = '/global/cscratch1/sd/zhenyuan/mpi_svm_python/data24/'
    main_folder = '/Users/zhenyuanliu/Dropbox/fall2017/EE227BT/project/mpi_svm_python/data4/'
    mylambda = 1.
    # pay attention, objval is the hinge loss!!!!!!!! not the objective value
    z, total_loss, objval, r_norm, s_norm, eps_pri, eps_dual = svm_admm(main_folder)
    objective_val = 0.5 * np.linalg.norm(z[0:-1, :])**2 + mylambda * total_loss[0]
    print('objective_val:' + str(objective_val))
    print('norm of w_hat is: ' + str(np.linalg.norm(z[0:-1, 0])))
    print('norm of b_hat is: ' + str(z[-1, 0]))

    # sio.savemat('z1.mat', {'z':z})


def svm_admm(main_folder, mylambda = 1., rho = 1., max_iter = 200, abs_tol = 1e-4, rel_tol = 1e-3):
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
    '''
    MPI
    '''
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    N = size

    '''
    Read in Data
    '''
    print('reading X' + str(rank) + '.npy')
    filename_X = main_folder + 'X' + str(rank) + '.npy'
    X = np.load(filename_X)
    filename_y = main_folder + 'y' + str(rank) + '.npy'
    y = np.load(filename_y)
    mX,nX = X.shape
    y_raveld = y.ravel() 
    # A is a matrix given by [-y_j*x_j -y_j]
    A = - np.dot(np.diag(y_raveld), np.concatenate((X, np.ones((mX, 1))), axis = 1))

    #Data preprocessing
    mA, nA = A.shape

    if rank == 0:
        tic = time.time()

    #initialize ADMM solver
    x = np.zeros((nA,1))
    z = np.zeros((nA,1))
    u = np.zeros((nA,1))
    r = np.zeros((nA,1))

    send = np.zeros(3)
    recv = np.zeros(3)
    local_loss = np.zeros(1)
    total_loss = np.zeros(1)

    # Saving state
    if rank == 0:
        print('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' %('iter',
                                                      'r norm', 
                                                      'eps pri', 
                                                      's norm', 
                                                      'eps dual', 
                                                      'objective'))
    objval     = []
    r_norm     = []
    s_norm     = []
    eps_pri    = []
    eps_dual   = []

    '''
    ADMM solver loop
    '''
    for k in range(max_iter):

        # u-update
        u = u + x - z

        # x-update 
        temp1 = z.reshape((z.shape[0], 1))
        temp2 = u.reshape((u.shape[0], 1))
        x_var = Variable(nA)
        constraints = []
        objective = Minimize(sum_entries(pos( A * x_var + 1)) + rho/2. * sum_squares((x_var - temp1 + temp2)))
        prob = Problem(objective, constraints)
        result = prob.solve()
        x_temp = x_var.value
        x_temp = x_temp.reshape((x_temp.shape[0], 1))
        x = x_temp # nA by 1

        send[0] = r.T.dot(r)[0][0] # sum ||r_i||_2^2
        send[1] = x.T.dot(x)[0][0] # sum ||x_i||_2^2
        send[2] = u.T.dot(u)[0][0]/(rho**2) # sum ||y_i||_2^2

        zprev = np.copy(z)
        
        w = x + u # w would be sent to calculate z
        comm.Barrier()
        comm.Allreduce([w, MPI.DOUBLE], [z, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([send, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM)
        
        # z-update
        z = N * rho/(1./mylambda + N * rho) * z * 1./N
        
        # diagnostics, reporting, termination checks
        local_loss[0] = hinge_loss(A, x)
        comm.Barrier()
        comm.Allreduce([local_loss, MPI.DOUBLE], [total_loss, MPI.DOUBLE], op=MPI.SUM)
        #
        objval.append(mylambda * total_loss[0] + 0.5 * np.linalg.norm(z[0:-1, :])**2)

        # prior residual -> norm(x-z)
        r_norm.append(np.sqrt(recv[0]))
        # dual residual -> norm(-rho*(z-zold))
        s_norm.append(np.sqrt(N) * rho * np.linalg.norm(z - zprev))
        eps_pri.append(np.sqrt(nA * N) * abs_tol + 
                       rel_tol * np.maximum(np.sqrt(recv[1]), np.sqrt(N) * np.linalg.norm(z)))
        eps_dual.append(np.sqrt(nA * N) * abs_tol + rel_tol * np.sqrt(recv[2]))

        if rank == 0:
            print('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' %(k+1,\
                                                              r_norm[k],\
                                                              eps_pri[k],\
                                                              s_norm[k],\
                                                              eps_dual[k],\
                                                              objval[k]))

        if r_norm[k] < eps_pri[k] and s_norm[k] < eps_dual[k] and k > 0:
            break

        #Compute primal residual
        r = x - z
        # print('r:' + str(norm(r)**2))

    if rank == 0:
        toc = time.time() - tic
        print("\nElapsed time is %.2f seconds"%toc)
        '''
        Store results
        '''
        np.save(main_folder + '/z', z)
        np.save(main_folder + '/objval', objval)
        np.save(main_folder + '/r_norm', r_norm)
        np.save(main_folder + '/s_norm', s_norm)
        np.save(main_folder + '/eps_pri', eps_pri)
        np.save(main_folder + '/eps_dual', eps_dual)
    return z, total_loss, objval, r_norm, s_norm, eps_pri, eps_dual




def myobjective(A, mylambda, x, z):
    return 0.5 * np.linalg.norm(z)**2 + mylambda * hinge_loss(A, x)

def hinge_loss(A, x):
    val = 0
    val = val + np.sum(np.maximum(np.dot(A, x.reshape((x.shape[0], 1))) + 1, np.zeros((A.shape[0], 1))))
    return val

if __name__=='__main__':
    main()
