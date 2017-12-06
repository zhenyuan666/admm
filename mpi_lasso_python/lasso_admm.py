import pdb,time
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky
import matplotlib.pyplot as plt

"""

"""
def main():
    # now solve the problem
    main_folder = '/global/cscratch1/sd/zhenyuan/mpi_lasso_python/data1/'
    main_folder = '/Users/zhenyuanliu/Dropbox/fall2017/EE227BT/project/mpi_lasso_python/data1/'
    i = 0
    filename_A = main_folder + 'A' + str(i) + '.npy'
    A = np.load(filename_A)
    filename_b = main_folder + 'b' + str(i) + '.npy'
    b = np.load(filename_b)
    #
    z, h = lasso_admm(A, b, 0.5)
    #print('objective value is : ' + str(h['objval'][-1]))
    print('norm of x_hat is: ' + str(norm(z)))

    K = len(h['objval'][np.where(h['objval']!=0)])

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    ax.plot(np.arange(K), h['objval'][:K],'k',ms=10,lw=2)
    ax.set_ylabel('f(x^k) + g(z^k)')
    ax.set_xlabel('iter (k)')

    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(211)
    ax1.semilogy(np.arange(K),np.maximum(1e-8,h['r_norm'][:K]),'k',lw=2)
    ax1.semilogy(np.arange(K),h['eps_pri'][:K],'k--',lw=2)
    ax1.set_ylabel('||r||_2')

    ax2 = fig2.add_subplot(212)
    ax2.semilogy(np.arange(K),np.maximum(1e-8,h['s_norm'][:K]),'k',lw=2)
    ax2.semilogy(np.arange(K),h['eps_dual'][:K],'k--',lw=2)
    ax2.set_ylabel('||s||_2')
    ax2.set_xlabel('iter (k)')

    plt.show()


def lasso_admm(A, b, mylambda, rho=1., rel_par=1., QUIET = False, MAX_ITER = 500, ABSTOL = 1e-6, RELTOL = 1e-4):
    """
     Solve lasso problem via ADMM
    
     [z, history] = lasso_admm(A, b, mylambda, rho, rel_par)
    
     Solves the following problem via ADMM:
    
       minimize 1/2*|| Ax - b ||_2^2 + mylambda || A ||_1
    
     The solution is returned in the vector z.
    
     history is a dictionary containing the objective value, the primal and
     dual residual norms, and the tolerances for the primal and dual residual
     norms at each iteration.
    
     rho is the augmented Lagrangian parameter.
    
     rel_par is the over-relaxation parameter (typical values for rel_par are
     between 1.0 and 1.8).
    
     More information can be found in the paper linked at:
     http://www.stanford.edu/~bobd/papers/distr_opt_stat_learning_admm.html
    """
    if not QUIET:
        tic = time.time()

    #Data preprocessing
    m, n = A.shape
    Atb = A.T.dot(b)

    #ADMM solver
    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    u = np.zeros((n, 1))

    # cache the (Cholesky) factorization
    L,U = factor(A, rho)

    if not QUIET:
        print('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' %('iter',
                                                      'r norm', 
                                                      'eps pri', 
                                                      's norm', 
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
        q = Atb + rho * (z - u) #(temporary value)
        if m >= n: # skinny matrix, use inverse of (A^T A + \rho I)
            x = spsolve(U,spsolve(L, q))[...,np.newaxis]
        else:
            ULAq = spsolve(U, spsolve(L, A.dot(q)))[...,np.newaxis]
            x = (q * 1./rho)-((A.T.dot(ULAq))*1./(rho**2))

        # z-update
        zold = np.copy(z)
        x_hat = rel_par * x + (1. - rel_par) * zold
        z = shrinkage(x_hat + u, mylambda * 1./rho)

        # u-update
        u = u + x_hat - z

        # diagnostics, reporting, termination checks
        h['objval'][k]   = objective(A, b, mylambda, x, z)
        h['r_norm'][k]   = norm(x - z)
        h['s_norm'][k]   = norm(rho * (z - zold))
        h['eps_pri'][k]  = np.sqrt(n) * ABSTOL+ RELTOL * np.maximum(norm(x), norm(-z))
        h['eps_dual'][k] = np.sqrt(n) * ABSTOL + RELTOL * norm(rho * u)
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

def objective(A, b, mylambda, x, z):
    return .5 * norm(A.dot(x) - b)**2 + mylambda * norm(z, 1)

def shrinkage(A, kappa):
    return np.maximum(0., A - kappa) - np.maximum(0., -A - kappa)

def factor(A, rho):
    m,n = A.shape
    if m >= n:
       L = cholesky(A.T.dot(A)+ rho * sparse.eye(n))
    else:
       L = cholesky(sparse.eye(m) + 1./rho * (A.dot(A.T)))
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L,U

if __name__=='__main__':
    main()
