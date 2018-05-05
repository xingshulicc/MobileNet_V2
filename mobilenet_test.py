# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
"""
Created on Fri May  4 13:43:46 2018

@author: xingshuli
"""
import os
#from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image
from keras import backend as K

from MobileNet_v2 import MobileNetv2

#pre-parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # '1' or '0' GPU

img_height, img_width = 64, 64

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

batch_size = 32
epochs = 500

train_data_dir = os.path.join(os.getcwd(), 'tiny_test/train')
validation_data_dir = os.path.join(os.getcwd(), 'tiny_test/validation')

num_classes = 25
nb_train_samples = 10000
nb_validation_samples = 2500

model = MobileNetv2(input_shape = input_shape, k = num_classes)
#optimizer = SGD(lr = 0.01, momentum = 0.9, decay = 1e-6, nesterov = True)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.summary()


train_datagen = ImageDataGenerator(rescale=1. / 255, 
                                   rotation_range=90, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

#set early-stopping parameters
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')

#set callbacks for model fit
callbacks = [early_stopping]

#model fit
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples //batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples //batch_size, 
    callbacks=callbacks)

#print acc and stored into acc.txt
f = open('/home/xingshuli/Desktop/acc.txt','w')
f.write(str(hist.history['acc']))
f.close()
#print val_acc and stored into val_acc.txt
f = open('/home/xingshuli/Desktop/val_acc.txt','w')
f.write(str(hist.history['val_acc']))
f.close()
#print val_loss and stored into val_loss.txt   
f = open('/home/xingshuli/Desktop/val_loss.txt', 'w')
f.write(str(hist.history['val_loss']))
f.close()

#the reasonable accuracy of model should be calculated based on
#the value of patience in EarlyStopping: accur = accur[-patience + 1:]/patience
Er_patience = 11  # Er_patience = patience + 1
accur = []
with open('/home/xingshuli/Desktop/val_acc.txt','r') as f1:
    data1 = f1.readlines()
    for line in data1:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        accur.append(num_float)
f1.close()

y = sum(accur, [])
ave = sum(y[-Er_patience:]) / len(y[-Er_patience:])
print('Validation Accuracy = %.4f' % (ave))











