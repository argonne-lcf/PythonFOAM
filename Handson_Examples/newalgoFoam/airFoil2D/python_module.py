print('Hello from python module')

import numpy as np
import matplotlib.pyplot as plt
print('You do not need to recompile for adding python functionality!')

def another_func(a):
    return 2*a


def my_func(a):
    print('This is function my_func')
    print('The sum of the input array is',np.sum(a))
    print(a.shape)

    b = another_func(a)

    print('This is the sum of b=2*a:',np.sum(b))

    plt.figure()
    plt.plot(a[:,0])
    plt.savefig('Myplot.png')
    plt.close()

    sumval = np.asarray([np.sum(a).astype('float64')]) # Very very important - return data as array and make sure type is appropriate
    return sumval