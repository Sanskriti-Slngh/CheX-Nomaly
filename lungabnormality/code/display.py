import pydicom
import os
import random
import numpy as np
from skimage.transform import resize
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# seed is 10000
random.seed(10000)

# variables
file_pids = "D:/tiya2023/lungabnormality/data/alldata.dat"
directory = 'F:/lungabnormality/data'
image_data_csv = 'D:/tiya2023/lungabnormality/data/train.csv/train.csv'

train_df = pd.read_csv(image_data_csv)


count = 0
count2 = 0
# iterating through data
for filename in os.listdir(directory):
    l = False
    f = os.path.join(directory, filename)

    # file id
    id = filename.split('.dicom')[0]

    # get info from csv on this file
    class_rows = train_df.loc[train_df['image_id'] == id]

    for index in range(len(class_rows)):
        disease = class_rows.iloc[index]['class_name']
        if disease != "No finding":
            l = True

    if l:
        count2 += 1
    else:
        count += 1

print(count)
print(count2)
print(count+count2)