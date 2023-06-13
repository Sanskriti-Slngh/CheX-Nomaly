# import libaries
import pickle
import pandas as pd
import pydicom as dicom
import os
import random

# seed is 10000
random.seed(10000)

# variables
directory = 'F:/lungabnormality/data'
image_data_csv = 'D:/tiya2023/lungabnormality/data/train.csv/train.csv'

final_ids_sep = {"Aortic enlargement": [],
                 "Atelectasis": [],
                 "Calcification": [],
                 "Cardiomegaly": [],
                 "Consolidation": [],
                 "ILD": [],
                 "Infiltration": [],
                 "Lung Opacity": [],
                 "Nodule-Mass": [],
                 "Other lesion": [],
                 "Pleural effusion": [],
                 "Pleural thickening": [],
                 "Pneumothorax": [],
                 "Pulmonary fibrosis": [],
                 "No finding": []}

final_ids_tog = []

# read csv
train_df = pd.read_csv(image_data_csv)

# iterating through data
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    # checking if it is a file
    if not os.path.isfile(f):
        print(f)
        exit()

    # file id
    id = filename.split('.dicom')[0]

    # get info from csv on this file
    class_row = train_df.loc[train_df['image_id'] == id]
    for index in range(len(class_row)):
        disease = class_row.iloc[index]['class_name']
        if disease == 'Nodule/Mass':
            disease = 'Nodule-Mass'

        if id not in final_ids_sep[disease]:
            final_ids_sep[disease].append(id)
            final_ids_tog.append(id)

# add no finding ids to seperate datasets
for disease in final_ids_sep:
    if disease == "No finding":
        pass
    else:
        count = len(final_ids_sep[disease])
        new_ids = random.sample(final_ids_sep['No finding'], count)
        for new_id in new_ids:
            final_ids_sep[disease].append(new_id)


print(final_ids_sep['ILD'])
print(final_ids_tog)


# randomizing ids
for disease in final_ids_sep:
    random.shuffle(final_ids_sep[disease])

random.shuffle(final_ids_tog)

# split train/val/test SEPARATE
# 80-10-10
for d in final_ids_sep:
    v = int(0.8 * len(final_ids_sep[d]))
    v1 = int(0.9 * len(final_ids_sep[d]))
    train = final_ids_sep[d][:v]
    val = final_ids_sep[d][v:v1]
    test = final_ids_sep[d][v1:]
    # os.makedirs("D:/tiya2023/lungabnormality/data/diseases")
    #file = open("D:/tiya2023/lungabnormality/data/diseases/" + d + ".dat", 'wb')
    #print(len(train), len(val), len(test))
    #pickle.dump((train, val, test), file)
    #file.close()

# split train/val/test TOGETHER
# 80-10-10
v = int(0.8 * len(final_ids_tog))
v1 = int(0.9 * len(final_ids_tog))
train = final_ids_tog[:v]
val = final_ids_tog[v:v1]
test = final_ids_tog[v1:]

print(len(train), len(val), len(test))

file = open("D:/tiya2023/lungabnormality/data/alldata.dat", 'wb')
pickle.dump((train, val, test), file)
file.close()