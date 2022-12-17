print('*****************here***************************')

import traceback
import sys
import matplotlib.pyplot as plt
import numpy as np

print('*****************here***************************')

def ml_nut(array):
    
    '''
    Array is np array with columns as variables
    '''
    print('*****************here***************************')
    print('Computing nut from Python')
  
    return (0.09*(array[:,0]**2)/(array[:,1]+1.0e-8)).reshape(-1,1).astype('double')


if __name__ == '__main__':
    print('This is the Python module for an ML Turbulence model')