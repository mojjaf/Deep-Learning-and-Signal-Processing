# -*- coding: utf-8 -*-
"""ResNet_module.ipynb

Created on Wed Nov 13 16:11:12 2019

@author: Mojtaba Jafaritadi, Ph.D.
"""

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras import regularizers 
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform,he_normal
import scipy.misc
from matplotlib.pyplot import imshow
from keras.callbacks import  Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Input, ZeroPadding1D, Dropout, LSTM,CuDNNLSTM,GRU
from keras.layers.convolutional import Conv1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.regularizers import l1
from keras import optimizers
from keras.layers import TimeDistributed
import keras.backend as K


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block for RESNET
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = Conv1D(filters=F1, kernel_size=1,strides =1,padding='same',input_shape=(None,n_timesteps,n_features), name = conv_name_base + '2a')(X)
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)   
    # Second component of main path (≈3 lines)
    X = Conv1D(filters=F2, kernel_size=f,strides =1,padding='same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)
    # Third component of main path (≈2 lines)
    X = Conv1D(filters=F3, kernel_size=1, strides =1,padding='same', name = conv_name_base + '2c')(X)
    X = BatchNormalization(name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X


def maxpool_block_1(X, f, filters, s, stage, block):
    """
    Implementation of the identity block for RESNET
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    mp_name_base = 'mp' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = Conv1D(filters=F1, kernel_size=f,strides =s,padding='same',input_shape=(None,n_timesteps,n_features), name = conv_name_base + '2a', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)
        
    # Second component of main path ()
    X = Conv1D(filters=F2, kernel_size=1, strides=1, name = conv_name_base + '2b', kernel_initializer = he_normal(seed=0))(X)
    
    X_shortcut = Conv1D(filters=F2, kernel_size=1, strides=1, name = conv_name_base + '1', kernel_initializer = he_normal(seed=0))(X_shortcut)

    X_shortcut = MaxPooling1D(pool_size=s, padding='same',name = mp_name_base + '1')(X_shortcut)

    

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X


def maxpool_block_2(X, f, filters, s, stage, block):
    """
    Implementation of the identity block for RESNET
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    mp_name_base = 'mp' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2 = filters
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)        
    # Second component of main path (≈3 lines)
    X = Conv1D(filters=F1, kernel_size=f,strides =s,padding='same',input_shape=(None,n_timesteps,n_features), name = conv_name_base + '2b', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)
    # Third component of main path (≈2 lines)
    X = Conv1D(filters=F2, kernel_size=1, strides =1,padding='same', name = conv_name_base + '2c', kernel_initializer = he_normal(seed=0))(X)
   

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv1D(filters=F2, kernel_size=1, strides=1, name = conv_name_base + '1', kernel_initializer = he_normal(seed=0))(X_shortcut)

    X_shortcut = MaxPooling1D(pool_size=s, padding='same',name = mp_name_base + '1')(X_shortcut)
   # print(X_shortcut.shape)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def convolutional_block(X, f, filters, stage, block,s=2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv1D(filters=F1, kernel_size=f,strides = s, padding='same', input_shape=(None,n_timesteps,n_features),name = conv_name_base + '2a', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)

    # Second component of main path (≈3 lines)
    X = Conv1D(filters=F2, kernel_size=f,strides = 1, padding='same', name = conv_name_base + '2b', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)

    # Third component of main path (≈2 lines)
    X = Conv1D(filters=F3, kernel_size=1,strides = 1,padding='same', name = conv_name_base + '2c', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2c')(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv1D(filters=F3, kernel_size=1,strides = s,padding='same', name = conv_name_base + '1', kernel_initializer = he_normal(seed=0))(X_shortcut)
   # print(X_shortcut.shape)
    X_shortcut = BatchNormalization(name = bn_name_base + '1')(X_shortcut)
    #X_shortcut = Dropout(0.25)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
        
    return X

def ResNet1D(input_shape = (256,6), classes = 2):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding1D(0)(X_input)
    
    # Stage 1
    X = Conv1D(filters=16, kernel_size=7,padding='same', name = 'conv1', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv1')(X)
    X = Activation('relu')(X)

    # Stage 2

    X = maxpool_block_1(X, f = 5, filters = [16, 32], s=1,stage = 2, block='a')
    X = identity_block(X, 3,  [16, 16,  32], stage=2, block='b')
    X = identity_block(X, 3,  [16, 16,  32], stage=2, block='c')
    X = convolutional_block(X, f = 2, filters = [16, 16, 32], s=1, stage = 2, block='d')

    # Stage 3 

    X = maxpool_block_2(X, 5, filters = [16, 32], s=2, stage=3, block='a')
    X = maxpool_block_2(X, 5, filters = [16, 32],s=1, stage=3, block='b')
    X = maxpool_block_2(X, 5, filters = [16, 32],s=2, stage=3, block='c')
    X = maxpool_block_2(X, 5, filters = [16, 32],s=2, stage=3, block='d')
    X = identity_block(X, 3,  [16, 16,  32], stage=3, block='e')
    X = identity_block(X, 3,  [16, 16,  32], stage=3, block='f')
    X = convolutional_block(X, f = 2, filters = [16, 16, 32], s=1,stage = 3, block='g')

    # Stage 4 
    X = maxpool_block_2(X, 3, filters = [32, 64],s=1, stage=4, block='a')
    X = maxpool_block_2(X, 3, filters = [32, 64], s=2,stage=4, block='b')
    X = maxpool_block_2(X, 5, filters = [32, 64],s=1, stage=4, block='c')
    X = maxpool_block_2(X, 5, filters = [32, 64], s=2,stage=4, block='d')
    X = identity_block(X, 5,  [32, 32,  64], stage=4, block='e')
    X = identity_block(X, 5,  [32, 32,  64], stage=4, block='f')
    X = convolutional_block(X, f = 2, filters = [32, 32, 64], s=1,stage = 4, block='g')

 # Stage 5
    X = maxpool_block_2(X, 5, filters = [64, 128],s=1, stage=5, block='a')
    X = maxpool_block_2(X, 5, filters = [64, 128], s=2,stage=5, block='b')
    X = maxpool_block_2(X, 5, filters = [64, 128], s=1,stage=5, block='c')
    X = maxpool_block_2(X, 5, filters = [64, 128],s=2, stage=5, block='d')
    X = identity_block(X, 3,  [64, 64,  128], stage=5, block='e')
    X = identity_block(X, 3,  [64, 64,  128], stage=5, block='f')
    X = convolutional_block(X, f = 2, filters = [64, 64,  128], s=1,stage = 5, block='g')
    
# Stage 6
    X = maxpool_block_2(X, 5, filters = [128, 256],s=1, stage=6, block='a')
    X = maxpool_block_2(X, 5, filters = [128, 256],s=2, stage=6, block='b')
    X = maxpool_block_2(X, 5, filters = [128, 256],s=1, stage=6, block='c')
    X = maxpool_block_2(X, 5, filters = [128, 256],s=2, stage=6, block='d')
    X = identity_block(X, 3,  [128, 128,  256], stage=6, block='e')
    X = identity_block(X, 3,  [128, 128,  256], stage=6, block='f')
    X = convolutional_block(X, f = 2, filters = [128, 128, 256], s=1,stage = 6, block='g')

    X = BatchNormalization(name = 'bn_final')(X)
    X = Activation('relu')(X)
    
    #X=LSTM(50, return_sequences=True,input_shape=(X.shape[1],1))(X)
    X = Dropout(0.2)(X)
    #X=LSTM(20)(X)
    #X = MaxPooling1D(pool_size=2, name='max_pool')(X)
   # X = TimeDistributed(Flatten())(X)
    X = Flatten()(X)
    #X = LSTM(100)(X)
   # X = CuDNNLSTM(1000)(X)
  
    #
    #X=Dense(10, activation='relu',activity_regularizer=l1(0.0001), kernel_regularizer=regularizers.l2(0.001))(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet1D')

    return model

model = ResNet1D(input_shape = (256, 6), classes = 2)


"""
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
%matplotlib inline

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

########################

model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

checkpoint = ModelCheckpoint(os.path.join(output_directory , 
"model_checkpoint_iter_" + str(r) +
".ckpt"), 
monitor= 'val_acc' ,verbose=0,
save_best_only=True,
mode='max',
save_weights_only=True)

csv_logger = CSVLogger(os.path.join(output_directory , 
"model_historylog_iter_" + str(r) +
".csv"), 
append=False)

model.fit(X_train, Y_train, validation_data= (X_val, Y_val),
epochs=200,
batch_size= batch_num, 
callbacks=[csv_logger,checkpoint], shuffle =
True, 
verbose=1)
model.save_weights(os.path.join(output_directory ,
"model_weights_iter_"
 + str(r)
 + ".h5”))
"""