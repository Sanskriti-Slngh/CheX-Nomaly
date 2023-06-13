import matplotlib.pyplot as plt
import pydicom
import pandas as pd
import numpy as np
from lungabnormality.code.model.DataGen import CustomDataGen
import os
import pickle
from skimage.transform import resize
from tensorflow.keras.models import load_model




diseases = {"Aortic enlargement": True,
                 "Atelectasis": False,
                 "Calcification": False,
                 "Cardiomegaly": False,
                 "Consolidation": False,
                 "ILD": False,
                 "Infiltration": False,
                 "Lung Opacity": False,
                 "Nodule-Mass": False,
                 "Other lesion": False,
                 "Pleural effusion": False,
                 "Pleural thickening": False,
                 "Pneumothorax": False,
                 "Pulmonary fibrosis": True,
                 "No finding": False}

diseases_pid = {"Aortic enlargement": '7c22cee85ef4ace76782964772819043',
                 "Atelectasis": 'e9581123b6819b2cd1bcf6ed35481520',
                 "Calcification": 'e31be972e181987a8600a8700c1ebe88',
                 "Cardiomegaly": '01cbbeab94b4d2bfd5cd8a467fee46a7',
                 "Consolidation": 'bd3fe876153eeddad8bab49b129ea081',
                 "ILD": 'd46fcfc88827c952da48421ecdac7e30',
                 "Infiltration": '8b704237f088b80ad737f48ff49d8cd9',
                 "Lung Opacity": 'f59ccb89d776a68b79292bef810333ac',
                 "Nodule-Mass": 'ea0ab2737896670ca5d52dd4b10285ab',
                 "Other lesion": '9d9caa9e06ec349f19f871e3fe2f343a',
                 "Pleural effusion": '9390e4ee9fcf6bdba3b1f03d40bfd4d1',
                 "Pleural thickening": 'eedb247767ce38e4dd2498c5dd56dc11',
                 "Pneumothorax": 'b11960f23db725bb4ba6f6741586a5f7',
                 "Pulmonary fibrosis": '863d09f3ec5bc88cfa15138876a20ab8',
                 "No finding": ''}

for i in diseases:
    if diseases[i] == True:
        disease = i

# to get pids
if False:
    file = open("D:/tiya2023/lungabnormality/data/diseases/" + disease + ".dat", 'rb')
    train_PIDs, val_PIDs, test_PIDs = pickle.load(file)
    file.close()
    CustomDataGen(val_PIDs[:10], 'D:/tiya2023/lungabnormality/data/train.csv/train.csv', 'F:/lungabnormality/data/', disease, batch_size=1, data_set='val', create=False)

#model = unet.Model(history, data_name, disease, all)
#model.compile(tf.keras.optimizers.Adam(learning_rate=lr, decay=0.001))
model = load_model("D:/tiya2023/lungabnormality/model/" + disease + ".h5", custom_objects={'dice_coef': dice_coef})

file = 'F:/lungabnormality/data/' + diseases_pid[disease] + '.dicom'
ds = pydicom.dcmread(file).pixel_array
width = ds.shape[0]
height = ds.shape[1]
ds1 = resize(ds, (512, 512, 1))
ds1 = np.reshape(ds1, (1, 512, 512, 1))
prediction = model.predict(ds1) > 0.5
prediction = np.reshape(prediction, (512, 512, 1))
plt.imshow(prediction)
plt.show()

# image = CustomDataGen(diseases_pid[disease], 'D:/tiya2023/lungabnormality/data/train.csv/train.csv', 'F:/lungabnormality/data/', disease, batch_size=1, data_set='val', create=False)
#y = self.model.predict(image)
#print(y)