import numpy as np


print('You have imported the python module successfully')

def adv_diff(mat,rhs):
    print(np.max(np.linalg.solve(mat,rhs)))
    print('Solution Size \n')
    print(np.shape(rhs)[0])
    return None

#mat = np.zeros((3))

#(adv_diff(mat))