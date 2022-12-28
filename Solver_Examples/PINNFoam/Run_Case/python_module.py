print('Module for training a neural network approximation to PDE')

import traceback
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print('***************** Python imports completed in PYthon module *******************************')

class pinn_model(object):
    """
    docstring for online_svd_calculator:
    K : Number of modes to truncate
    ff : Forget factor
    """
    def __init__(self, num_layers, num_neurons):
        super(pinn_model, self).__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons

    def initialize(self):
        # Construct a new neural network and compile (no training here)
        self.model = tf.keras.models.Sequential()

        # Input layer
        self.model.add(tf.keras.layers.Dense(units=self.num_neurons, activation='relu', input_dim=2)) # Coordinate inputs

        # Hidden layers
        for i in range(self.num_layers):
            self.model.add(tf.keras.layers.Dense(units=self.num_neurons,activation='relu'))

        # Output
        self.model.add(tf.keras.layers.Dense(units=4)) # Field variables output

        # Compile
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='mean_squared_error',optimizer=opt)


    def train_model(self,x,y):
        # Train model here
        self.model.fit(x,y,epochs=1000,batch_size=256)


    def deploy_model(self,x):
        return self.model.predict(x).numpy()
        # Predict from inputs

iter = 0
training_data = None
pinn_model = pinn_model(5,40)
pinn_model.initialize()

def snapshot_func(array,rank):

    global iter, training_data

    if iter == 0:
        print('Collecting snapshots iteration: ',iter)
        
        # array = np.expand_dims(array,axis=0)
        training_data = array.copy()

        iter+=1
    else:
        print('Collecting snapshots iteration: ',iter)
        
        training_data = np.concatenate((training_data,array),axis=0)

        iter+=1

    return 0

def pinn_train_func(rank):
    
    global iter, training_data
    global pinn_model

    input_data = training_data[:,:2]
    output_data = training_data[:,2:]

    pinn_model.train_model(input_data,output_data)

    training_data = None

    return 0

def pinn_deploy_func(array,rank):
    
    global pinn_model

    return np.asarray(pinn_model.deploy_model(array).numpy(),dtype='float64')


if __name__ == '__main__':
    print('This is the Python module for PINNFoam')