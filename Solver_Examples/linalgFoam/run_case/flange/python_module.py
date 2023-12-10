print('Entered Python Module')
import numpy as np
from scipy import sparse
import sys, petsc4py
import matplotlib.pyplot as plt

petsc4py.init(sys.argv)
from petsc4py import PETSc


counter = 0

res1 = np.array([])
res2 = np.array([])
res3 = np.array([])

def petsc2array(v):
    s = v.getArray()
    return s



def adv_diff(mat,rhs,sol):
    global counter
    global res1,res2,res3
    print('Entered solver function')
    ksp = PETSc.KSP().create()
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.JACOBI)
    n = np.shape(rhs)[0]
    x = PETSc.Vec().createSeq(n)
    b = PETSc.Vec().createSeq(n)
    b.setValues(range(n),rhs)

    Ascipy = sparse.csr_matrix(mat)
    
    A = PETSc.Mat().createAIJWithArrays(Ascipy.shape, (Ascipy.indptr, Ascipy.indices, Ascipy.data))

    ksp.setOperators(A)
    ksp.solve(b, x)
    
    print('Converged in', ksp.getIterationNumber(), 'iterations.') 
    sol_petsc= petsc2array(x)

    sol_python = ((np.linalg.solve(mat,rhs)))
    print('L2 norm between OpenFOAM and NumPy solution')
    print(np.linalg.norm(sol-sol_python))
    print('L2 norm between OpenFOAM and PETSc solution')
    print(np.linalg.norm(sol-sol_petsc))


    PETSc_residual = np.mean(rhs-np.dot(mat,sol_petsc))
    NumPy_residual = np.mean(rhs-np.dot(mat,sol_python))
    OpenFOAM_residual = np.mean(rhs-np.dot(mat,sol))
    counter = counter+1
    
    res1 = np.append(res1,PETSc_residual)
    res2 = np.append(res2,NumPy_residual)
    res3 = np.append(res3,OpenFOAM_residual)
    
    print(np.shape(res1))
   
    if counter > 1799:

        plt.plot(res1,label = 'PETSc')
        plt.plot(res2,label = 'NumPy')
        plt.plot(res3,label = 'OpenFOAM')
        
        
        if counter > 1799:
           plt.legend()
           plt.savefig("Residual_Plot.png")

    return None

#mat = np.zeros((3))

#adv_diff(mat))
#adv_diff(np.random.rand(3,3),rhs=np.ones((3,1)),sol=np.zeros((3,1)))