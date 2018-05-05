# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu May  3 15:27:48 2018

@author: xingshuli
"""
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.applications.mobilenet import relu6, DepthwiseConv2D

from keras import backend as K

def _conv_block(inputs, filters, kernel, strides):
    '''
    Convolutional block includes 2D convolution operation with Batchnorm and relu6
    Arguments:
    filters: the number of output channels
    kernel: tuple of 2 integers, the kernel_size for Conv2D
    strides: tuple of 2 integers, specifying the strides of Conv2D
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = Conv2D(filters, kernel, strides = strides, padding = 'same')(inputs)
    x = BatchNormalization(axis = channel_axis)(x)
    x = Activation(relu6)(x)
    
    return x
    
def _bottleneck(inputs, filters, kernel, t, s, r = False):
    '''
    Linear bottleneck for inverted residual block below
    Arguments:
    filters: the number of output channels
    kernel: tuple of 2 integers, the kernel_size for DepthwiseConv2D
    t: Integer, expansion factor 
    s: An integer, specifying the strides of depthwise convolution
    
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    
    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
    
    x = DepthwiseConv2D(kernel, strides = (s, s), depth_multiplier = 1, padding = 'same')(x)
    x = BatchNormalization(axis = channel_axis)(x)
    x = Activation(relu6)(x)
    
    x = Conv2D(filters, (1, 1), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization(axis = channel_axis)(x)
    
    if r:
        x = add([x, inputs])
    
    return x 
        
    
def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    '''
    Inverted residual block as referred in paper 'MobileNetV2'
    all layers in the same sequence have the same number of output channels, the first
    layer of each sequence has a stride s and all others use stride 1
    We should note that the first layer in each sequence cannot use residual connection
    even for strides = 1, because the channels of input tensor is not equal to that of output.
    
    Arguments:
    filters: the number of output channels
    kernel: tuple of 2 integers, the kernel_size for depthwise convolution
    t: Integer, expansion factor
    strides: An integer, specifying the strides of depthwise convolution
    n: Integer, layer repeat times
    '''
    x = _bottleneck(inputs, filters, kernel, t, strides)
    
    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)
    
    return x
    
def MobileNetv2(input_shape, k):
    '''
    Arguments:
    input_shape: shape of input tensor
    We should note that the resolution of input image cannot be 
    lower than 32 by 32  
    k: Integer, number of classes
    '''
    inputs = Input(shape = input_shape)
    x = _conv_block(inputs, 32, (3, 3), (2, 2))
    
    x = _inverted_residual_block(x, 16, (3, 3), t = 1, strides = 1, n = 1)
    x = _inverted_residual_block(x, 24, (3, 3), t = 6, strides = 2, n = 2)
    x = _inverted_residual_block(x, 32, (3, 3), t = 6, strides = 2, n = 3)
    x = _inverted_residual_block(x, 64, (3, 3), t = 6, strides = 2, n = 4)
    x = _inverted_residual_block(x, 96, (3, 3), t = 6, strides = 1, n = 3)
    x = _inverted_residual_block(x, 160, (3, 3), t = 6, strides = 2, n = 3)
    x = _inverted_residual_block(x, 320, (3, 3), t = 6, strides = 1, n = 1)
    
    x = _conv_block(x, 1280, (1, 1), (1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1280))(x)
    x = Conv2D(k, (1, 1), padding = 'same')(x)
    
    x = Activation('softmax')(x)
    outputs = Reshape((k,))(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    return model

    









