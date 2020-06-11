import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from keras.models import Model
from keras.optimizers import adam

def resnet_skip(x1, x2):
    return x1 + x2

def vnet():
    #Encoding
    inputs = Input(shape = (128,128,128,1))

    conv1_1 = Conv3D(16, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(inputs)
    conv1_2 = Conv3D(16, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv1_1)
    skip1 = resnet_skip(conv1_2, inputs)

    pool1 = MaxPooling3D(pool_size = (2,2,2))(skip1)

    conv2_1 = Conv3D(32, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(pool1)
    conv2_2 = Conv3D(32, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv2_1)
    skip2 = resnet_skip(pool1, conv2_2)

    pool2 = MaxPooling3D(pool_size = (2,2,2))(skip2)

    conv3_1 = Conv3D(64, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(pool2)
    conv3_2 = Conv3D(64, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv3_1)
    conv3_3 = Conv3D(64, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv3_2)
    skip3 = resnet_skip(pool2, conv3_3)

    pool3 = MaxPooling3D(pool_size = (2,2,2))(skip3)

    conv4_1 = Conv3D(128, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(pool3)
    conv4_2 = Conv3D(128, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv4_1)
    conv4_3 = Conv3D(128, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv4_2)
    skip4 = resnet_skip(pool3, conv4_3)

    pool4 = MaxPooling3D(pool_size = (2,2,2))(skip4)

    conv5_1 = Conv3D(256, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(pool4)
    conv5_2 = Conv3D(256, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv5_1)
    conv5_3 = Conv3D(256, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv5_2)
    skip5 = resnet_skip(pool4, conv5_3)

    #Decoding
    upsample1 = UpSampling3D(size = (2,2,2), interpolation = 'nearest')(skip5)
    concat1 = concatenate([conv4_3, upsample1], axis = 4)

    conv6_1 = Conv3D(128, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(concat1)
    conv6_2 = Conv3D(128, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv6_1)
    conv6_3 = Conv3D(128, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv6_2)
    skip6 = resnet_skip(upsample1, conv6_3)

    upsample2 = UpSampling3D(size = (2,2,2), interpolation = 'nearest')(skip6)
    concat2 = concatenate([conv3_3, upsample2], axis = 4)

    conv7_1 = Conv3D(64, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(concat2)
    conv7_2 = Conv3D(64, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv7_1)
    conv7_3 = Conv3D(64, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv7_2)
    skip7 = resnet_skip(upsample2, conv7_3)

    upsample3 = UpSampling3D(size = (2,2,2), interpolation = 'nearest')(skip7)
    concat3 = concatenate([conv2_2, upsample3], axis = 4)

    conv8_1 = Conv3D(32, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(concat3)
    conv8_2 = Conv3D(32, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv8_1)
    skip8 = resnet_skip(upsample3, conv8_2)

    upsample4 = UpSampling3D(size = (2,2,2), interpolation = 'nearest')(skip8)
    concat4 = concatenate([conv1_2, upsample4], axis = 4)

    conv9_1 = Conv3D(16, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(concat4)
    conv9_2 = Conv3D(16, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv9_1)
    conv9_3 = Conv3D(2, (1,1,1), strides = 1, padding = 'same', activation = 'relu')(conv9_2)
    conv10 = Conv3D(1, (1,1,1), strides = 1, padding = 'same', activation = 'sigmoid')(conv9_3)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = vnet()
print(model.summary())