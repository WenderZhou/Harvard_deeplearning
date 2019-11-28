import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Add,Dense, Activation,AveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal

def CNN_block(X, f, filters, blockname, stride = 1):
    """
    ResNet50 building block's CNN counterpart
    
    Arguments:
    X -- input images or feature maps
    f -- integer, specifying the shape f*f of middle conv2d layer
    filters -- tuple, specifying the number of filters in the block
    blockname -- string, defining the name of the block
    stride -- integer, specifying the stride of the first layer and shortcut connection
    
    Return:
    output -- output of the CNN block
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters

    # first conv2D layer
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (s, s), padding = 'valid', 
               name = blockname + '_conv2d_1', 
               kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = blockname + '_bn_1')(X)
    X = Activation('relu')(X)
    
    # second conv2D layer
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', 
               name = blockname + '_conv2d_2', 
               kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = blockname + '_bn_2')(X)
    X = Activation('relu')(X)

    # third conv2D layer
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', 
               name = blockname + '_conv2d_3', 
               kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = blockname + '_bn_3')(X)
    output = Activation('relu')(X)
    
    return output

def CNN50(input_shape=(64, 64, 3), classes=6):
    """
    ResNet50's CNN counterpart
    
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=he_normal(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = CNN_block(X, f=3, filters=(64, 64, 256), blockname="conv2_1", shortcut="projection")
    X = CNN_block(X, f=3, filters=(64, 64, 256), blockname="conv2_2", shortcut="identity")
    X = CNN_block(X, f=3, filters=(64, 64, 256), blockname="conv2_3", shortcut="identity") 

    # Stage 3
    X = CNN_block(X, f=3, filters=(128, 128, 512), blockname="conv3_1", shortcut="projection", s=2)
    X = CNN_block(X, f=3, filters=(128, 128, 512), blockname="conv3_2", shortcut="identity")
    X = CNN_block(X, f=3, filters=(128, 128, 512), blockname="conv3_3", shortcut="identity") 
    X = CNN_block(X, f=3, filters=(128, 128, 512), blockname="conv3_4", shortcut="identity")

    # Stage 4
    X = CNN_block(X, f=3, filters=(256, 256, 1024), blockname="conv4_1", shortcut="projection", s=2)
    X = CNN_block(X, f=3, filters=(256, 256, 1024), blockname="conv4_2", shortcut="identity")
    X = CNN_block(X, f=3, filters=(256, 256, 1024), blockname="conv4_3", shortcut="identity") 
    X = CNN_block(X, f=3, filters=(256, 256, 1024), blockname="conv4_4", shortcut="identity")
    X = CNN_block(X, f=3, filters=(256, 256, 1024), blockname="conv4_5", shortcut="identity") 
    X = CNN_block(X, f=3, filters=(256, 256, 1024), blockname="conv4_6", shortcut="identity")

    # Stage 5
    X = CNN_block(X, f=3, filters=(512, 512, 2048), blockname="conv5_1", shortcut="projection", s=2)
    X = CNN_block(X, f=3, filters=(512, 512, 2048), blockname="conv5_2", shortcut="identity")
    X = CNN_block(X, f=3, filters=(512, 512, 2048), blockname="conv5_3", shortcut="identity") 

    # AVGPOOL
    X = AveragePooling2D((2,2), name="avg_pool")(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = he_normal(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='CNN50')

    return model