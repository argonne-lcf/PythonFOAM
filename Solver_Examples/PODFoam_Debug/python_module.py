# f = open('python_log_file','w')
# f.write('Starting python module from OpenFOAM')
# f.close()

print('************************** here *****************************')

import traceback
import sys
import matplotlib.pyplot as plt
import numpy as np

print('************************** here *****************************')

class online_svd_calculator(object):
    """
    docstring for online_svd_calculator:
    K : Number of modes to truncate
    ff : Forget factor
    """
    def __init__(self, K, ff):
        super(online_svd_calculator, self).__init__()
        self.K = K
        self.ff = ff

    def initialize(self,A):
        # Computing R-SVD of the initial matrix - step 1 section II
        q, r = np.linalg.qr(A)

        # Compute SVD of r - v is already transposed  - step 2 section II
        # https://stackoverflow.com/questions/24913232/using-numpy-np-linalg-svd-for-singular-value-decomposition
        ui, self.di, self.vit = np.linalg.svd(r) 

        # Get back U and truncate
        self.ui = np.matmul(q,ui)[:,:self.K]  #- step 3 section II
        self.di = self.di[:self.K]

    def incorporate_data(self,A):
        """
        A is the new data matrix
        """
        # Section III B 3(a):
        m_ap = self.ff*np.matmul(self.ui,np.diag(self.di))
        m_ap = np.concatenate((m_ap,A),axis=-1)
        udashi, ddashi = np.linalg.qr(m_ap)

        # Section III B 3(b):
        utildei, dtildei, vtildeti = np.linalg.svd(ddashi)

        # Section III B 3(c):
        max_idx = np.argsort(dtildei)[::-1][:self.K]
        self.di = dtildei[max_idx]
        utildei = utildei[:,max_idx]
        self.ui = np.matmul(udashi,utildei)

iter = 0
u_snapshots = None
v_snapshots = None
w_snapshots = None

online_mode = True
init_mode = True
u_svd_calc = online_svd_calculator(5,0.95)
v_svd_calc = online_svd_calculator(5,0.95)
w_svd_calc = online_svd_calculator(5,0.95)

def method_of_snapshots(snapshot_matrix):
    # Get rid of snapshot mean
    smean = np.mean(snapshot_matrix,axis=1)
    snapshot_fluc = snapshot_matrix-smean[:,None]

    new_mat = np.matmul(np.transpose(snapshot_fluc),snapshot_fluc)
    w,v = np.linalg.eig(new_mat)
    # Bases
    V = np.real(np.matmul(snapshot_fluc,v)) 
    trange = np.arange(np.shape(V)[1])
    V[:,trange] = V[:,trange]/(1.0e-6+np.sqrt(w[:])) # Avoid divide by zero

    # Truncate according to L2 reconstruction error
    truncation = 5
    # variance = 0.0
    # while variance < 0.95:
    #     variance = np.sum(w[0:truncation])/np.sum(w)
    #     truncation+=1

    return_data = V[:,:truncation]

    return return_data

def snapshot_func(array,rank):

    global iter, u_snapshots, v_snapshots, w_snapshots

    # if rank == 0:
    #     print("Python ux values from rank 0:",array[:5,0])
    #     print("Python uy values from rank 0:",array[:5,1])
    #     print("Python uz values from rank 0:",array[:5,2])

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

    return 0

def svd_func(rank):
    
    global online_mode
    global iter, u_snapshots, v_snapshots, w_snapshots
    global init_mode, u_svd_calc, v_svd_calc, w_svd_calc

    if online_mode:
        
        if init_mode:
            print('Performing online SVD on snapshots rankwise - initialization')
            u_svd_calc.initialize(u_snapshots)
            v_svd_calc.initialize(v_snapshots)
            w_svd_calc.initialize(w_snapshots)

            u_modes = u_svd_calc.ui
            v_modes = v_svd_calc.ui
            w_modes = w_svd_calc.ui

            init_mode = False
        else:
            u_svd_calc.incorporate_data(u_snapshots)
            v_svd_calc.incorporate_data(v_snapshots)
            w_svd_calc.incorporate_data(w_snapshots)

            u_modes = u_svd_calc.ui
            v_modes = v_svd_calc.ui
            w_modes = w_svd_calc.ui

            print('New data incorporated - sending back to OpenFOAM')
    
    else:    
        u_modes = method_of_snapshots(u_snapshots)
        v_modes = method_of_snapshots(v_snapshots)
        w_modes = method_of_snapshots(w_snapshots)

        print('SVD finished - sending back to OpenFOAM')
  
    u_snapshots = None
    v_snapshots = None
    w_snapshots = None

    iter = 0

    return_data = np.concatenate((u_modes,v_modes,w_modes),axis=0)
    
    # if rank == 0:
    #     print("Python values d0: ",return_data[0:5,0])
    #     print("Python values d1: ",return_data[0,0:5]) # Direction of array being read into a pointer in C++

    # if rank == 0:
    #     num_cells = np.shape(u_modes)[0]
    #     print("Python Ux modal values: ",return_data[0:5,2])
    #     print("Python Uy modal values: ",return_data[num_cells:num_cells+5,2])
    #     print("Python Uz modal values: ",return_data[2*num_cells:2*num_cells+5,2])

    return return_data

if __name__ == '__main__':
    print('This is the Python module for PODFoam')