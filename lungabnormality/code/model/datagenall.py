# import libraries
import numpy as np
import tensorflow as tf
import pickle
import bz2
import random

# create custom data generator
class CustomDataGenAll(tf.keras.utils.Sequence):

    def __init__(self, batch_size, data_set):

        # define variables
        self.batch = batch_size
        self.data_set = data_set
        self.size = 512

        is_file = "D:/tiya2023/lungabnormality/data/allvinbig" + self.data_set + ".pbz2"

        print("Getting saved data for all diseases")
        data = bz2.BZ2File(is_file)
        self.x, self.y = pickle.load(data)
        print("got data")

       # self.x = self.x[:10]
       # self.y = self.y[:10]

        self.n = len(self.y)

        self.x_2 = self.x.copy()
        self.y_2 = self.y.copy()

        count = 0
        for ind, mask in enumerate(self.y_2):
            if np.sum(mask) > 0:
                count += 1
                if data_set == 'train':
                    for i in range(10):
                        self.x.append(self.x_2[ind])
                        self.y.append(mask)


        print("Amount of positive: " + str(count*11))
        print("Amount of negative: " + str(len(self.y)-count*11))
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

        # normalize data using min-max method
        x = np.array(x)
        y = np.array(y)

        x = np.reshape(x, (x.shape[0], self.size, self.size, 1))
        y = np.reshape(y, (x.shape[0], self.size, self.size, 1))
        return x, y

    def __len__(self):
        return self.n // self.batch