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

# create custom data generator
class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, pids, csv, dir, disease,
                 batch_size, data_set, aug, create=True):

        # define variables
        self.pids = pids[:]
        self.n = len(self.pids)
        self.image_data_csv = csv
        self.main_directory = dir
        self.disease = disease
        self.batch = batch_size
        self.data_set = data_set
        self.augumentation = aug
        self.size = 512
        is_file = 'D:/tiya2023/lungabnormality/data/' + self.disease + self.data_set + '.pbz2'

        if os.path.isfile(is_file) and create:
            print("Getting saved data for " + self.disease)
            data = bz2.BZ2File(is_file)
            self.x, self.y = pickle.load(data)

            if self.augumentation:
                self.x_2 = self.x.copy()
                self.y_2 = self.y.copy()

                for ind, mask in enumerate(self.y_2):
                    if np.sum(mask) > 0:
                        for i in range(50):
                            self.x.append(self.x_2[ind])
                            self.y.append(mask)

            print(len(self.y))


        else:
            df = pd.read_csv(self.image_data_csv)
            self.x = []
            self.y = []
            for pid in self.pids:
                mask = np.zeros((self.size, self.size, 1))
                filename = self.main_directory + pid + ".dicom"
                ds = pydicom.dcmread(filename).pixel_array

                # print(ds.shape)

                # print(ds[100, 100], ds[100, 101], ds[100, 102], ds[100, 103])

                # print(np.min(ds), np.max)

                image = resize(ds, (self.size, self.size, 1), preserve_range=True)
                min_value = np.min(image)
                max_value = np.max(image)
                image = image - min_value
                image = (image / (max_value-min_value)).astype(np.float16)

                # print(image[100, 100], image[100, 101], image[100, 102], image[100, 103])

                # print(np.min(image), np.max(image))


                width = ds.shape[0]
                height = ds.shape[1]

                class_row = df.loc[df['image_id'] == pid]

                for index in range(len(class_row)):
                    disease = class_row.iloc[index]['class_name']
                    if disease == "Nodule/Mass":
                        disease = "Nodule-Mass"

                    if disease == self.disease:
                        x_min = class_row.iloc[index]['x_min']
                        x_max = class_row.iloc[index]['x_max']
                        y_min = class_row.iloc[index]['y_min']
                        y_max = class_row.iloc[index]['y_max']

                        bb = [int(x_min * self.size / height), int(y_min * self.size / width), int(x_max * self.size / height),
                              int(y_max * self.size / width)]
                        mask[bb[1]:bb[3], bb[0]:bb[2], :] = 1



                #plt.imshow(np.reshape(image, (512, 512)))
                #plt.contour(np.reshape(mask, (512, 512)))
                #plt.show()

                self.x.append(image)
                self.y.append(mask)

                if np.sum(mask) > 0 and False:
                    for i in range(50):
                        self.x.append(image)
                        self.y.append(mask)

                if False:
                    i = np.flipud(image)
                    m = np.flipud(mask)

                    # vertical flip
                    self.x.append(i)
                    self.y.append(m)


                    # rotation
                    for angle in [5, 10, 15, 20, 25]:
                        i = ndimage.rotate(image, angle, reshape=False)
                        m = ndimage.rotate(mask, angle, reshape=False) > 0.5
                        self.x.append(np.flipud(i))
                        self.y.append(np.flipud(m))

                    # gaussion filter - 3
                    for i in [1.05, 1.1, 1.15]:
                        i = self.gaussian_noise(image, i)
                        self.x.append(i)
                        self.y.append(mask)

                    # contrast
                    for m1 in [-50, -25, 25, 50]:
                        i = image + m1
                        i[i > 255] = 255
                        i[i < 0] = 0
                        self.x.append(i)
                        self.y.append(mask)


            # saving data into zipped file
            # with bz2.BZ2File('D:/tiya2023/lungabnormality/data/' + self.disease + self.data_set + '.pbz2', 'w') as f:
            #         pickle.dump((self.x, self.y), f)

            print(len(self.y))

    def gaussian_noise(self, img, mean=0, sigma=0.03):
        img = img.copy()
        noise = np.random.normal(mean, sigma, img.shape)
        mask_overflow_upper = img + noise >= 1.0
        mask_overflow_lower = img + noise < 0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0
        img += noise
        return img

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