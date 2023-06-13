# import libraries
import numpy as np
import tensorflow as tf
import pydicom
import pickle
import bz2
import os
import pandas as pd
from skimage.transform import resize
import  matplotlib.pyplot as plt
from scipy import ndimage
import random
from tensorflow.keras.models import load_model


# create custom data generator
class CustomDataGenSpecial(tf.keras.utils.Sequence):

    def __init__(self, data_set, aug, create=True):

        # define variables
        self.data_set = data_set
        self.augumentation = aug
        self.size = 512
        self.batch = 8
        is_file ='D:/tiya2023/lungabnormality/data/allvinbig' + self.data_set + '.pbz2'

        if os.path.isfile(is_file) and create:
            print("Getting saved data")
            data = bz2.BZ2File(is_file)
            self.x, self.y = pickle.load(data)
            print("got")

            #self.x = self.x[:20]
            # self.y = self.y[:20]

            self.n = len(self.y)
            print(len(self.y))

    def __getitem__(self, idx):
        if idx == 0:
            indices = [i for i in range(len(self.x))]
            random.shuffle(indices)
            xxx = [self.x[i] for i in indices]
            yyy = [self.y[i] for i in indices]
            self.x = xxx
            self.y = yyy

            del xxx, yyy

        if (idx + 1) * self.batch > len(self.x):
            x = self.x[idx * self.batch: len(self.x)]
            y = self.y[idx * (self.batch): len(self.x)]
        else:
            x = self.x[idx * self.batch: (idx + 1) * self.batch]
            y = self.y[idx * (self.batch): (idx + 1) * self.batch]

        preds = []

        for d in ['Aortic Enlargement', 'Atelectasis', 'Calcification' , 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity',
            'Nodule-Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']:
            n = "D:/tiya2023/lungabnormality/model/" + d
            m = load_model(n + '.h5')
            x_l = np.array(x)
            x_l = np.reshape(x_l, (x_l.shape[0], self.size, self.size, 1))
            a = m.predict(x_l, batch_size=2)
            preds.append(a)

        preds = np.array(preds)

        # normalize data using min-max method
        x = (np.max(preds, axis=0))
        y = np.array(y)

        x = np.reshape(x, (x.shape[0], self.size, self.size, 1))
        y = np.reshape(y, (x.shape[0], self.size, self.size, 1))
        return x, y

    def __len__(self):
        return self.n // self.batch