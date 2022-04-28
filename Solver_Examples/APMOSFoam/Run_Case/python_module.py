import traceback
import sys
import matplotlib.pyplot as plt
import numpy as np
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

iter = 0
u_snapshots = None
v_snapshots = None
w_snapshots = None
num_modes = 5 # This should match truncation

def snapshot_func(array,rank):

    global iter, u_snapshots, v_snapshots, w_snapshots

    if iter == 0:
        print('Collecting snapshots iteration: ',iter)
        
        u_snapshots = array[:,0].reshape(-1,1)
        v_snapshots = array[:,1].reshape(-1,1)
        w_snapshots = array[:,2].reshape(-1,1)

        iter+=1
    else:
        print('Collecting snapshots iteration: ',iter)
        
        u_temp = array[:,0].reshape(-1,1)
        v_temp = array[:,1].reshape(-1,1)
        w_temp = array[:,2].reshape(-1,1)

        u_snapshots = np.concatenate((u_snapshots,u_temp),axis=-1)
        v_snapshots = np.concatenate((v_snapshots,v_temp),axis=-1)
        w_snapshots = np.concatenate((w_snapshots,w_temp),axis=-1)

        iter+=1

# Method of snapshots to accelerate
def generate_right_vectors_mos(Y):
    '''
    Y - Snapshot matrix - shape: NxS
    returns V - truncated right singular vectors
    '''
    new_mat = np.matmul(np.transpose(Y),Y)
    w, v = np.linalg.eig(new_mat)

    svals = np.sqrt(np.abs(w))
    # rval = np.argmax(svals<0.0001) # eps0
    rval = 50

    return v[:,:rval].astype('double'), np.sqrt(np.abs(w[:rval])).astype('double') # Covariance eigenvectors, singular values

def apmos_func(placeholder):
    
    global iter, u_snapshots, v_snapshots, w_snapshots # Iteration and local data

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    snapshots_list = [u_snapshots, v_snapshots, w_snapshots]
    phi_list = []

    for local_data in snapshots_list:
        local_data_mean = np.mean(local_data,axis=1)
        local_data = local_data-local_data_mean[:,None]

        # Run a method of snapshots
        vlocal, slocal = generate_right_vectors_mos(local_data)

        # Find W
        wlocal = np.matmul(vlocal,np.diag(slocal).T)

        print('Mean vlocal is: ',np.mean(vlocal),'from rank: ',rank)
        print('Mean slocal is: ',np.mean(slocal),'from rank: ',rank)

        # Gather data at rank 0:
        wglobal = comm.gather(wlocal,root=0)

        # perform SVD at rank 0:
        if rank == 0:
            temp = wglobal[0]
            for i in range(nprocs-1):
                temp = np.concatenate((temp,wglobal[i+1]),axis=-1)
            wglobal = temp

            x, s, y = np.linalg.svd(wglobal)
        else:
            x = None
            s = None
        
        x = comm.bcast(x,root=0)
        s = comm.bcast(s,root=0)

        # Find truncation threshold
        rval = num_modes

        # perform APMOS at each local rank
        phi_local = []
        for mode in range(rval):
            phi_temp = 1.0/s[mode]*np.matmul(local_data,x[:,mode:mode+1])
            phi_local.append(phi_temp)

        temp = phi_local[0]
        for i in range(rval-1):
            temp = np.concatenate((temp,phi_local[i+1]),axis=-1)
        
        phi_list.append(temp)
   
    print('APMOS finished - sending back to OpenFOAM')

    print('Shape of things in python ************')
    print(np.mean(phi_list))

    del u_snapshots, v_snapshots, w_snapshots
    del snapshots_list

    u_snapshots = None
    v_snapshots = None
    w_snapshots = None

    iter = 0

    return_data = np.concatenate((phi_list[0],phi_list[1],phi_list[2]),axis=0)

    return return_data

if __name__ == '__main__':
    print('This is the Python module for APMOSFoam')