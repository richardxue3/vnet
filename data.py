import h5py
import tensorflow as tf
import numpy as np
import random

class generator():
    def __init__(self,directory, train_batch_size, test_batch_size):
        f = h5py.File(directory,'r')
        self.dataset = list(f[key] for key in f.keys())
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_set = self.dataset[0:110]
        self.test_set = self.dataset[110:]

    def traingenerator(self):
        batch_size = self.train_batch_size
        trainset = self.train_set
        total_size = len(trainset)
        while True:
            for i in range(total_size//batch_size):
                batch_data_list = list(trainset[i]['volume'] for i in range(i*batch_size,(i+1)*batch_size))
                batch_masks_list = list(trainset[i]['segmentation'] for i in range(i*batch_size,(i+1)*batch_size))
                train_batch_data = np.reshape(batch_data_list,[batch_size, 128, 128, 128, 1])/255.0
                segmentation = np.reshape(batch_masks_list,[batch_size, 128, 128, 128, 1])
                train_batch_mask = (segmentation>128).astype(np.float)
                yield (train_batch_data, train_batch_mask)
            # Shuffles the train dataset for another epoch
            random.shuffle(trainset)

    def testgenerator(self):
        batch_size = self.test_batch_size
        testset = self.test_set
        total_size = len(testset)
        while True:
            for i in range(total_size//batch_size):
                batch_data_list = list(testset[i]['volume'] for i in range(i*batch_size,(i+1)*batch_size))
                batch_masks_list = list(testset[i]['segmentation'] for i in range(i*batch_size,(i+1)*batch_size))
                test_batch_data = np.reshape(batch_data_list,[batch_size, 128, 128, 128, 1])/255.0
                segmentation = np.reshape(batch_masks_list,[batch_size, 128, 128, 128, 1])
                test_batch_mask = (segmentation>128).astype(np.float)
                yield (test_batch_data, test_batch_mask)
            # Shuffles the train dataset for another epoch
            random.shuffle(testset)
    def eval(self,i):
        batch_size = self.test_batch_size
        testset = self.test_set
        data = testset[i]
        batch_data_list = data['volume']
        batch_masks_list = data['segmentation']
        test_batch_data = np.reshape(batch_data_list,[batch_size, 128, 128, 128, 1])/255.0
        segmentation = np.reshape(batch_masks_list,[batch_size, 128, 128, 128, 1])
        test_batch_mask = (segmentation>128).astype(np.float)
        return (test_batch_data, test_batch_mask)

class sliver_generator():
    def __init__(self,directory, train_batch_size, test_batch_size):
        f = h5py.File(directory,'r')
        self.dataset = list(f[key] for key in f.keys())
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def testgenerator(self):
        batch_size = self.test_batch_size
        testset = self.dataset
        total_size = len(testset)
        while True:
            for i in range(total_size//batch_size):
                batch_data_list = list(testset[i]['volume'] for i in range(i*batch_size,(i+1)*batch_size))
                batch_masks_list = list(testset[i]['segmentation'] for i in range(i*batch_size,(i+1)*batch_size))
                test_batch_data = np.reshape(batch_data_list,[batch_size, 128, 128, 128, 1])
                segmentation = np.reshape(batch_masks_list,[batch_size, 128, 128, 128, 1])
                test_batch_mask = (segmentation>128).astype(np.float)
                yield (test_batch_data, test_batch_mask)
            # Shuffles the train dataset for another epoch
            random.shuffle(testset)

       

