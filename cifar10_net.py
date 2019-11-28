import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Add,Dense, Activation,AveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal

def building_block(X, filters, kernels, blockname, shortcut, s = 1):
    """
    building block with shortcut connection perform an identity mapping or projection
    
    Arguments:
    X -- input images or feature maps
    filters -- tuple, specifying the number of filters in the block
    kernels -- tuple, specifying the shape of conv2d layer
    blockname -- string, defining the name of the block
    shortcut -- string, 'identity' or 'projection', deciding the function of shortcut connnecting
    s -- integer, using a stride of 2 to match dimension, otherwise 1
    
    Return:
    output -- output of the building block
    """
    
    # Retrieve Filters
    F1, F2 = filters
    k1, k2 = kernels
    
    # shortcut perform the identity mapping
    if(shortcut == 'identity'):
        X_shortcut = X
    # shortcut perform the projection mapping, use a stride(s) of 2 to match different size
    elif(shortcut == 'projection'):
        X_shortcut = Conv2D(filters = F2, kernel_size = (1, 1), strides = (s,s), padding = 'same', 
                            name = blockname + '_conv2d_shortcut',
                            kernel_initializer = he_normal(seed=0))(X)
        X_shortcut = BatchNormalization(axis = 3, name = blockname + '_bn_shortcut')(X_shortcut)
        
        
    # first component of main path
    X = Conv2D(filters = F1, kernel_size = (k1, k1), strides = (s, s), padding = 'same', 
               name = blockname + '_conv2d_1', 
               kernel_initializer = he_normal(seed=0),
              kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization(axis = 3, name = blockname + '_bn_1')(X)
    X = Activation('relu')(X)
    
    # second component of main path
    X = Conv2D(filters = F2, kernel_size = (k2, k2), strides = (1,1), padding = 'same', 
               name = blockname + '_conv2d_2', 
               kernel_initializer = he_normal(seed=0),
              kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization(axis = 3, name = blockname + '_bn_2')(X)
    
    X = Add()([X, X_shortcut])
    output = Activation('relu')(X)
    
    return output

def ResNet20(input_shape=(32, 32, 3), classes=10):
    """
    Implement the model ResNet20 mentioned in original paper
    
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Stage 2
    X = building_block(X_input, filters=(16, 16), kernels=(3,3), blockname="conv2_1", shortcut="projection")
    X = building_block(X, filters=(16, 16), kernels=(3,3), blockname="conv2_2", shortcut="identity")
    X = building_block(X, filters=(16, 16), kernels=(3,3), blockname="conv2_3", shortcut="identity")

    # Stage 3
    X = building_block(X, filters=(32, 32), kernels=(3,3), blockname="conv3_1", shortcut="projection", s=2)
    X = building_block(X, filters=(32, 32), kernels=(3,3), blockname="conv3_2", shortcut="identity")
    X = building_block(X, filters=(32, 32), kernels=(3,3), blockname="conv3_3", shortcut="identity")

    # Stage 4
    X = building_block(X, filters=(64, 64), kernels=(3,3), blockname="conv4_1", shortcut="projection", s=2)
    X = building_block(X, filters=(64, 64), kernels=(3,3), blockname="conv4_2", shortcut="identity")
    X = building_block(X, filters=(64, 64), kernels=(3,3), blockname="conv4_3", shortcut="identity")


    # AVGPOOL
    X = AveragePooling2D(pool_size=8, name="avg_pool")(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = he_normal(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet')

    return model

def CNN_block(X, filters, kernels, blockname, shortcut, s = 1):
    """
    building block's CNN counterpart
    
    Arguments:
    X -- input images or feature maps
    filters -- tuple, specifying the number of filters in the block
    kernels -- tuple, specifying the shape of conv2d layer
    blockname -- string, defining the name of the block
    shortcut -- string, 'identity' or 'projection', deciding the function of shortcut connnecting
    s -- integer, using a stride of 2 to match dimension, otherwise 1
    
    Return:
    output -- output of the building block
    """
    
    # Retrieve Filters
    F1, F2 = filters
    k1, k2 = kernels   
        
    # first component of main path
    X = Conv2D(filters = F1, kernel_size = (k1, k1), strides = (s, s), padding = 'same', 
               name = blockname + '_conv2d_1', 
               kernel_initializer = he_normal(seed=0),
              kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization(axis = 3, name = blockname + '_bn_1')(X)
    X = Activation('relu')(X)
    
    # second component of main path
    X = Conv2D(filters = F2, kernel_size = (k2, k2), strides = (1,1), padding = 'same', 
               name = blockname + '_conv2d_2', 
               kernel_initializer = he_normal(seed=0),
              kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization(axis = 3, name = blockname + '_bn_2')(X)
    output = Activation('relu')(X)
    
    return output

def CNN20(input_shape=(32, 32, 3), classes=10):
    """
    Implement a counterpart of ResNet20
    
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Stage 2
    X = CNN_block(X_input, filters=(16, 16), kernels=(3,3), blockname="conv2_1", shortcut="projection")
    X = CNN_block(X, filters=(16, 16), kernels=(3,3), blockname="conv2_2", shortcut="identity")
    X = CNN_block(X, filters=(16, 16), kernels=(3,3), blockname="conv2_3", shortcut="identity")

    # Stage 3
    X = CNN_block(X, filters=(32, 32), kernels=(3,3), blockname="conv3_1", shortcut="projection", s=2)
    X = CNN_block(X, filters=(32, 32), kernels=(3,3), blockname="conv3_2", shortcut="identity")
    X = CNN_block(X, filters=(32, 32), kernels=(3,3), blockname="conv3_3", shortcut="identity")

    # Stage 4
    X = CNN_block(X, filters=(64, 64), kernels=(3,3), blockname="conv4_1", shortcut="projection", s=2)
    X = CNN_block(X, filters=(64, 64), kernels=(3,3), blockname="conv4_2", shortcut="identity")
    X = CNN_block(X, filters=(64, 64), kernels=(3,3), blockname="conv4_3", shortcut="identity")


    # AVGPOOL
    X = AveragePooling2D(pool_size=8, name="avg_pool")(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = he_normal(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet')

    return model