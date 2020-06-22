import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose, Concatenate, Add
from keras.models import Model
from keras.optimizers import adam

def resnet_skip(x1, x2):
    return Add()([x1, x2])

def vnet():
    #Encoding
    
    #128
    inputs = Input(shape = (128,128,128,1))

    conv1_1 = Conv3D(16, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(inputs)
    conv1_2 = Conv3D(16, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv1_1)
    skip1 = resnet_skip(conv1_2, inputs)

    down_conv1 = Conv3D(32, (3,3,3), strides = 2, activation = 'relu', padding = 'same')(skip1)

    #64
    conv2_1 = Conv3D(32, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(down_conv1)
    conv2_2 = Conv3D(32, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv2_1)
    skip2 = resnet_skip(down_conv1, conv2_2)

    down_conv2 = Conv3D(64, (3,3,3), strides = 2, activation = 'relu', padding = 'same')(skip2)

    #32
    conv3_1 = Conv3D(64, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(down_conv2)
    conv3_2 = Conv3D(64, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv3_1)
    conv3_3 = Conv3D(64, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv3_2)
    skip3 = resnet_skip(down_conv2, conv3_3)

    down_conv3 = Conv3D(128, (3,3,3), strides = 2, activation = 'relu', padding = 'same')(skip3)
    
    #16
    conv4_1 = Conv3D(128, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(down_conv3)
    conv4_2 = Conv3D(128, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv4_1)
    conv4_3 = Conv3D(128, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv4_2)
    skip4 = resnet_skip(down_conv3, conv4_3)

    down_conv4 = Conv3D(256, (3,3,3), strides = 2, activation = 'relu', padding = 'same')(skip4)

    #8
    conv5_1 = Conv3D(256, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(down_conv4)
    conv5_2 = Conv3D(256, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv5_1)
    conv5_3 = Conv3D(256, (3,3,3), strides = 1, activation = 'relu', padding = 'same')(conv5_2)
    skip5 = resnet_skip(down_conv4, conv5_3)

    #Decoding
    upconv1 = Conv3DTranspose(128, (2,2,2), strides = 2, activation = 'relu', padding = 'same')(skip5)
    concat1 = Concatenate(axis = 4)([conv4_3, upconv1])

    conv6_1 = Conv3D(128, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(concat1)
    conv6_2 = Conv3D(128, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv6_1)
    conv6_3 = Conv3D(128, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv6_2)
    skip6 = resnet_skip(upconv1, conv6_3)

    upconv2 = Conv3DTranspose(64, (2,2,2), strides = 2, activation = 'relu', padding = 'same')(skip6)
    concat2 = Concatenate(axis = 4)([conv3_3, upconv2])

    conv7_1 = Conv3D(64, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(concat2)
    conv7_2 = Conv3D(64, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv7_1)
    conv7_3 = Conv3D(64, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv7_2)
    skip7 = resnet_skip(upconv2, conv7_3)

    upconv3 = Conv3DTranspose(32, (2,2,2), strides = 2, activation = 'relu', padding = 'same')(skip7)
    concat3 = Concatenate(axis = 4)([conv2_2, upconv3])

    conv8_1 = Conv3D(32, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(concat3)
    conv8_2 = Conv3D(32, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv8_1)
    skip8 = resnet_skip(upconv3, conv8_2)

    upconv4 = Conv3DTranspose(16, (2,2,2), strides = 2, activation = 'relu', padding = 'same')(skip8)
    concat4 = Concatenate(axis = 4)([conv1_2, upconv4])

    conv9_1 = Conv3D(16, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(concat4)
    conv9_2 = Conv3D(16, (3,3,3), strides = 1, padding = 'same', activation = 'relu')(conv9_1)
    conv9_3 = Conv3D(16, (1,1,1), strides = 1, padding = 'same', activation = 'relu')(conv9_2)
    skip9 = resnet_skip(upconv4, conv9_3)
    conv10 = Conv3D(1, (1,1,1), strides = 1, padding = 'same', activation = 'sigmoid')(skip9)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = vnet()
print(model.summary())
