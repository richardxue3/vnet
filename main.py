from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from data import *
from model import *

generator = generator('lits_val.h5', train_batch_size = 4, test_batch_size = 1)
train_data = generator.traingenerator()
test_data = generator.testgenerator()

if os.path.exists('./logs') == False:
    os.mkdir('./logs')
if os.path.exists('./Model') == False:
    os.mkdir('./Model')

output_model_file = './Model/NomVnet.h5'

callbacks = [TensorBoard(log_dir='./logs'),
            ModelCheckpoint(output_model_file,save_best_only=True),
            EarlyStopping(patience = 5, min_delta=1e-3)]

model = vnet()

model.fit_generator(train_data, steps_per_epoch = 28, epochs = 30, validation_data = test_data, 
                    validation_steps = 28, verbose = 1, callbacks = callbacks)