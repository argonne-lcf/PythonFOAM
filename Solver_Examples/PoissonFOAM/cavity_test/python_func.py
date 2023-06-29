import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.optimizers import Adam


u_inputs = Input(shape=(1600))
v_inputs = Input(shape=(1600))
x = Dense(5, activation='tanh', dtype='float64')(u_inputs)
y = Dense(5, activation='tanh', dtype='float64')(v_inputs)
z = Concatenate(axis =1)([x, y])
z = Dense(1600, activation='tanh', dtype='float64')(z)
model = Model(inputs=[u_inputs,v_inputs],outputs=z)
model.summary()

lr = 1e-3
it =0
inv = 0

def velocity_inp(array,array_3,array_4,array_5):
    array = tf.Variable(array,dtype = tf.float64) #velocity array
    array_3 = tf.Variable(array_3,dtype = tf.float64) #A matrix n*n
    input_1 = tf.reshape(array[:,0],(1,1600)) #u vel
    input_2 = tf.reshape(array[:,1],(1,1600)) #v vel
    array_4= tf.reshape(array_4,(1600,1)) #B matrix
    array_5= tf.reshape(array_5,(1600,1))  # Matrix for Cell volume
    left = tf.math.multiply(array_4,array_5)
    global lr,it,inv
    if it%50==0:
       inv = tf.linalg.inv(array_3) #preconditioning the residual minimizer function f once every 50 simple iterations/ training iters#
    it = it+1 
    for iter in range(0,40):# internal iterations 
            with tf.GradientTape() as tape:
                output_1 = model([input_1/10,input_2/10],training = True) #output of the model (pressure)
                output_2 = tf.reshape(output_1,(1600,1))
                loss1 = tf.subtract(left,tf.matmul(array_3,output_2)) 
                loss2 = tf.reduce_mean(tf.abs(tf.matmul(inv,loss1)),axis = 1,keepdims = True) 
                if iter%5==0: 
                	print("Inter iteration = ",iter, "... Inter loss = ",np.max(tf.abs(loss2).numpy()))
                grads = tape.gradient(loss2, model.trainable_variables)
                Adam(learning_rate= lr).apply_gradients(zip(grads, model.trainable_variables))
            if iter%5==0:
               lr = lr/5
    output_1 = model([input_1/10,input_2/10])
    model.save("overfit_model")
    lr = 1e-3/(it*2) 
    return output_1.numpy().T
    














