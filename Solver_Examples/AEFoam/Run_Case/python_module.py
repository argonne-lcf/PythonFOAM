# f = open('log.python_module','w')
# f.write('Starting python module from OpenFOAM')
# f.close()

import traceback
import sys
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)

import tensorflow as tf
tf.random.set_seed(10)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model, Sequential, Model

# Custom activation (swish)
def my_swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)

# Some global variables
iter = 0
snapshots = None
model_num = 0

class autoencoder_model(object):
    """
    docstring for autoencoder_model:
    K : Latent space dimensionality
    """
    def __init__(self, num_latent):
        super(autoencoder_model, self).__init__()
        self.num_latent = num_latent
        self.num_epochs=1000
        self.batch_size=128
        self.lrate = 0.001

        print('Autoencoder object initialized')

    def train(self,A,rank):

        global model_num

        # # Open a text file for plotting encoded variables
        # text_file = open('Latent_Representation_'+str(rank)+'_model_num_'+str(model_num)+'.txt','w')
        # text_file.write('Encoded representation'+'\n')
        # text_file.close()

        print('Total size of training data:',A.shape)
        num_snapshots, num_points, num_fields = A.shape

        A = A.reshape(num_snapshots,-1)

        # Encoder
        encoder_inputs = Input(shape=(num_points*num_fields),name='Field')

        x = Dense(50, activation=my_swish)(encoder_inputs)
        x = Dense(25, activation=my_swish)(x)
        x = Dense(10, activation=my_swish)(x)
        encoded = Dense(self.num_latent)(x)
        self.encoder = Model(inputs=encoder_inputs,outputs=encoded)

        ## Decoder
        decoder_inputs = Input(shape=(self.num_latent,),name='decoded')
        x = Dense(10,activation=my_swish)(decoder_inputs)
        x = Dense(25,activation=my_swish)(x)
        x = Dense(50,activation=my_swish)(x)
        decoded = Dense(num_points*num_fields)(x)

        self.decoder = Model(inputs=decoder_inputs,outputs=decoded)

        ## Autoencoder
        ae_outputs = self.decoder(self.encoder(encoder_inputs))

        self.model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='Autoencoder')

        weights_filepath = 'weights_rank_'+str(rank)+'_time_'+str(model_num)+'.h5'

        my_adam = optimizers.Adam(lr=self.lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        callbacks_list = [checkpoint,earlystopping]

        # Compile network
        self.model.compile(optimizer=my_adam,loss='mean_squared_error')
        self.model.summary()

        # Train model
        # Shuffle data
        idx = np.arange(num_snapshots)
        np.random.shuffle(idx)
        A = A[idx]

        train_history = self.model.fit(x=A,
                              y=A,
                              epochs=self.num_epochs, batch_size=self.batch_size,
                              callbacks=callbacks_list, validation_split=0.1)

        model_num+=1

    def encode_field(self,A,rank):
        num_points, num_fields = A.shape
        A = A.reshape(1,-1)

        encoded = np.asarray(K.eval(self.encoder(A.astype('float32'))))
        print('Encoded representation:',encoded)

        # # Write out encoded flow field to text file
        # text_file = open('Latent_Representation_'+str(rank)+'model_num_'+str(model_num)+'.txt','a')
        # np.savetxt(text_file,encoded)
        # text_file.close()

        decoded = np.asarray(K.eval(self.decoder(encoded.astype('float32'))))

        print(np.mean(decoded),np.mean(A))

        print('Mean squared error of reconstruction:',np.sqrt(np.mean((decoded-A)**2)))

        return decoded.reshape(num_points,num_fields).astype('float64')




autoencoder_class_object = autoencoder_model(num_latent=4)

def snapshot_func(array,rank):

    global iter, snapshots

    if iter == 0:
        print('Collecting snapshots iteration: ',iter)
        snapshots = np.expand_dims(array,axis=0)
        iter+=1
    else:
        print('Collecting snapshots iteration: ',iter)

        array = np.expand_dims(array,axis=0)
        snapshots = np.concatenate((snapshots,array),axis=0)
        iter+=1

    return 0


def autoencoder_func(rank):
    global iter, snapshots
    global autoencoder_class_object

    autoencoder_class_object.train(snapshots,rank)

    snapshots = None
    iter = 0

    return None

def encode_func(array,rank):
    print('***********************************here**********************************')
    global autoencoder_class_object

    decoded = autoencoder_class_object.encode_field(array,rank)

    return decoded



if __name__ == '__main__':
    print('This is the Python module for AutoencoderFoam')
