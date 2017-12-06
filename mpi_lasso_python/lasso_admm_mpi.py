from __future__ import division
import pdb,time,os
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky
from mpi4py import MPI
import scipy.io as sio

"""

"""

def main():
    main_folder = '/global/cscratch1/sd/zhenyuan/mpi_lasso_python/data24/'
    main_folder = '/Users/zhenyuanliu/Dropbox/fall2017/EE227BT/project/mpi_lasso_python/data4/'
    z, total_loss, r_norm, s_norm, eps_pri, eps_dual = lasso_admm(main_folder)
    print('objective value is : ' + str(total_loss[0]))
    print('norm of x_hat is: ' + str(norm(z)))


def lasso_admm(main_folder, mylambda = .5, rho = 1., max_iter = 200, abs_tol = 1e-6, rel_tol = 1e-4):
    """
     Lasso problem:

       minimize 1/2*|| A x - b ||_2^2 + mylambda || x ||_1
    Output:
            - z      : solution of the Lasso problem
            - objval : objective value
            - r_norm : primal residual norm
            - s_norm : dual residual norm
            - eps_pri: tolerance for primal residual norm
            - eps_pri: tolerance for dual residual norm
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
    print('reading A' + str(rank) + '.npy')
    filename_A = main_folder + 'A' + str(rank) + '.npy'
    A = np.load(filename_A)
    filename_b = main_folder + 'b' + str(rank) + '.npy'
    b = np.load(filename_b)
    m,n = A.shape

    if rank == 0:
        tic = time.time()

    #save a matrix-vector multiply
    Atb = A.T.dot(b)
    
    #initialize ADMM solver
    x = np.zeros((n,1))
    z = np.zeros((n,1))
    u = np.zeros((n,1))
    r = np.zeros((n,1))

    send = np.zeros(3)
    recv = np.zeros(3)    
    local_loss = np.zeros(1)
    total_loss = np.zeros(1)

    # cache the (Cholesky) factorization
    L,U = factor(A, rho)

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
        u= u + x - z
        # print('u:' + str(norm(u)**2))
         
        # x-update 
        q = Atb + rho * (z - u) #(temporary value)

        if m >= n: # skinny matrix, use inverse of (A^T A + \rho I)
            x = spsolve(U,spsolve(L, q))[...,np.newaxis]
        else:
            ULAq = spsolve(U, spsolve(L, A.dot(q)))[...,np.newaxis]
            x = (q * 1./rho)-((A.T.dot(ULAq))*1./(rho**2))
        # print('x:' + str(norm(x)**2))

        send[0] = r.T.dot(r)[0][0] # sum ||r_i||_2^2
        send[1] = x.T.dot(x)[0][0] # sum ||x_i||_2^2
        send[2] = u.T.dot(u)[0][0]/(rho**2) # sum ||y_i||_2^2

        zprev = np.copy(z)
        
        w = x + u # w would be sent to calculate z
        comm.Barrier()
        comm.Allreduce([w, MPI.DOUBLE], [z, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([send, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM)
        
        # z-update
        z = soft_threshold(z * 1./N, mylambda * 1./(N * rho))
        # print('z:' + str(norm(z)**2))
        
        # diagnostics, reporting, termination checks
        local_loss[0] = objective(A, b, mylambda, z)
        comm.Barrier()
        comm.Allreduce([local_loss, MPI.DOUBLE], [total_loss, MPI.DOUBLE], op=MPI.SUM)
        #
        total_loss[0] = total_loss[0] - (N - 1) * mylambda * norm(z, 1)
        objval.append(total_loss[0])
        
        # prior residual -> norm(x-z)
        r_norm.append(np.sqrt(recv[0]))
        # dual residual -> norm(-rho*(z-zold))
        s_norm.append(np.sqrt(N) * rho * norm(z - zprev))
        eps_pri.append(np.sqrt(n * N) * abs_tol + 
                       rel_tol * np.maximum(np.sqrt(recv[1]), np.sqrt(N) * norm(z)))
        eps_dual.append(np.sqrt(n * N) * abs_tol + rel_tol * np.sqrt(recv[2]))

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
    return z, total_loss, r_norm, s_norm, eps_pri, eps_dual

def objective(A, b, mylambda, z):
    return .5 * norm(A.dot(z) - b)**2 + mylambda * norm(z, 1)

def factor(A, rho):
    m,n = A.shape
    if m >= n:
       L = cholesky(A.T.dot(A) + rho * sparse.eye(n))
    else:
       L = cholesky(sparse.eye(m) + 1./rho * (A.dot(A.T)))
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L,U

def soft_threshold(v, k):
    return np.maximum(0., v - k) - np.maximum(0., -v - k)

if __name__=='__main__':
    main()
