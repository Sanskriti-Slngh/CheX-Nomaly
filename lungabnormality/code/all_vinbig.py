# import libaries
import pickle
import pandas as pd
import pydicom
import os
import random
import numpy as np
from skimage.transform import resize
import  matplotlib.pyplot as plt


# seed is 10000
random.seed(10000)

# variables
file_pids = "D:/tiya2023/lungabnormality/data/alldata.dat"
directory = 'F:/lungabnormality/data'
image_data_csv = 'D:/tiya2023/lungabnormality/data/train.csv/train.csv'

file = open(file_pids, 'rb')
train_PIDs, val_PIDs, test_PIDs = pickle.load(file)
file.close()

train_df = pd.read_csv(image_data_csv)

data_x = []
data_y = []
train_x = []
train_y = []
val_x = []
val_y = []
test_x = []
test_y = []

# iterating through data
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    # file id
    id = filename.split('.dicom')[0]

    # get info from csv on this file
    class_rows = train_df.loc[train_df['image_id'] == id]

    ds = pydicom.dcmread(f).pixel_array

    image = resize(ds, (512, 512, 1), preserve_range=True)
    min_value = np.min(image)
    max_value = np.max(image)
    image = image - min_value
    image = (image / (max_value - min_value)).astype(np.float16)

    width = ds.shape[0]
    height = ds.shape[1]

    diseases = []
    mask = np.zeros((512, 512, 1))
    for index in range(len(class_rows)):
        disease = class_rows.iloc[index]['class_name']
        if disease=="No finding":
            q = False
        else:
            q = True
            x_min = class_rows.iloc[index]['x_min']
            x_max = class_rows.iloc[index]['x_max']
            y_min = class_rows.iloc[index]['y_min']
            y_max = class_rows.iloc[index]['y_max']

            bb = [int(x_min * 512 / height), int(y_min * 512 / width), int(x_max * 512 / height),
                  int(y_max * 512 / width)]
            mask[bb[1]:bb[3], bb[0]:bb[2], :] = 1

    if id in train_PIDs and q:
        print("train")
        train_x.append(image)
        train_y.append(mask)
    elif id in val_PIDs and q:
        print("val")
        val_x.append(image)
        val_y.append(mask)
    elif id in test_PIDs and q:
        print("test")
        test_x.append(image)
        test_y.append(mask)
    else:
        continue

print(len(train_y))
print(len(val_y))
print(len(test_y))

import bz2

# saving data into zipped file
with bz2.BZ2File('D:/tiya2023/lungabnormality/data/allvinbigtrainpositive.pbz2', 'w') as f:
    pickle.dump((train_x, train_y), f)

with bz2.BZ2File('D:/tiya2023/lungabnormality/data/allvinbigvalpositive.pbz2', 'w') as f:
    pickle.dump((val_x, val_y), f)

with bz2.BZ2File('D:/tiya2023/lungabnormality/data/allvinbigtestpositive.pbz2', 'w') as f:
    pickle.dump((test_x, test_y), f)