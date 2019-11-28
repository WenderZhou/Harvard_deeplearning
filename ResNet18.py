import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Add,Dense, Activation,AveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.models import Model
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
               kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = blockname + '_bn_1')(X)
    X = Activation('relu')(X)
    
    # second component of main path
    X = Conv2D(filters = F2, kernel_size = (k2, k2), strides = (1,1), padding = 'same', 
               name = blockname + '_conv2d_2', 
               kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = blockname + '_bn_2')(X)
    
    X = Add()([X, X_shortcut])
    output = Activation('relu')(X)
    
    return output

def ResNet18(input_shape=(64, 64, 3), classes=6):
    """
    Implement the model ResNet18 mentioned in original paper
    
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=he_normal(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = building_block(X, filters=(64, 64), kernels=(3,3), blockname="conv2_1", shortcut="projection")
    X = building_block(X, filters=(64, 64), kernels=(3,3), blockname="conv2_2", shortcut="identity")

    # Stage 3
    X = building_block(X, filters=(128, 128), kernels=(3,3), blockname="conv3_1", shortcut="projection", s=2)
    X = building_block(X, filters=(128, 128), kernels=(3,3), blockname="conv3_2", shortcut="identity")

    # Stage 4
    X = building_block(X, filters=(256, 256), kernels=(3,3), blockname="conv4_1", shortcut="projection", s=2)
    X = building_block(X, filters=(256, 256), kernels=(3,3), blockname="conv4_2", shortcut="identity")

    # Stage 5
    X = building_block(X, filters=(512, 512), kernels=(3,3), blockname="conv5_1", shortcut="projection", s=2)
    X = building_block(X, filters=(512, 512), kernels=(3,3), blockname="conv5_2", shortcut="identity")

    # AVGPOOL
    X = AveragePooling2D((2,2), name="avg_pool")(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = he_normal(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet18')

    return model