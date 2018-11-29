# MobileNet_V2
MobileNet_V2 is a good example of model compression


Paper Reference: MobileNetV2: Inverted Residuals and Linear Bottlenecks

Code Reference: https://github.com/xiaochus/MobileNetV2


1. How to import relu6 and DepthwiseConv2D in different version of Keras

For 2.0.7:

from keras.applications.mobilenet import relu6, DepthwiseConv2D

For 2.1.6

from keras.layers import DepthwiseConv2D

def relu6(x):

    return K.relu(x, max_value=6)

2. Citrus pests and diseases recognition

The weights of trained model are saved, which initialized with 'imagenet'. The validation accuracy is 96.87%.

The description about citrus pests and diseases can be found in:

https://github.com/xingshulicc/Citrus-Pests-Description

https://github.com/xingshulicc/Citrus-Diseases-Description


If you have questions, please send email to me: xingshuli600@gmail.com
