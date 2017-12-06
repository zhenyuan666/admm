from mpi4py import MPI  





comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print('hello world')  
print('my rank is: ' + str(rank))  
