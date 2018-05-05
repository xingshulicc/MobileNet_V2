# MobileNet_V2
MobileNet_V2 is a good example of model compression


Paper Reference: MobileNetV2: Inverted Residuals and Linear Bottlenecks

Code Reference: https://github.com/xiaochus/MobileNetV2

1. dataset description

I selected 25 classes of Tiny ImageNet as dataset for training and testing. Some data augmentation methods are also employed
to improve the model generalization.


2. Experiment

Device: GPU 1080 ti (11G)

Software: Keras (version 2.0.7) with tensorflow (GPU) backend.

During training, two optimizers are chosen for comparison. And in order to reduce the effect of overfitting, early-stopping is
added with 10 patience.

optimizer_1 = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

optimizer_2 = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)

In fact, I found that the validation accuracy of SGD is more stable than that of Adam. Therefore SGD is a better choice for training 
tiny images.

3. How to import relu6 and DepthwiseConv2D in different version of Keras

For 2.0.7:

from keras.applications.mobilenet import relu6, DepthwiseConv2D

For 2.1.6

from keras.layers import DepthwiseConv2D

def relu6(x):

    return K.relu(x, max_value=6)






