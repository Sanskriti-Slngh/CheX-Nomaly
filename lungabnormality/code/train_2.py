import pickle
import numpy as np
import bz2

disease = ['Aortic Enlargement', 'Atelectasis', 'Calcification' , 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity',
            'Nodule-Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
x_all = []
for d in disease:
    print(d)
    x = []
    for i in range(3):
        with bz2.BZ2File('D:/tiya2023/lungabnormality/data/y_hat' + d + str(i) + '.pbz2', 'r') as f:
            a = pickle.load(f)
            print("l")
            a = a.tolist()
            print("k")
        x.append(a)
        print(len(x))

    x = np.array(x)
    print(x.shape)
    x = np.reshape(x, (x.shape[0], 512, 512, 1))
    print(x.shape)

    with bz2.BZ2File('E:/d/y_hat' + d + '.pbz2', 'w') as f:
        pickle.dump((x), f)

    x_all.append(x)

x_all = np.array(x_all)
print(x_all.shape)
x_all = np.reshape(x_all, (x.shape[0], 512, 512, len(disease)))
print(x_all.shape)

is_file = 'D:/tiya2023/lungabnormality/data/allvinbigtrain.pbz2'
data = bz2.BZ2File(is_file)
_, y = pickle.load(data)

y = np.array(y)
y = np.reshape(y, (y.shape[0], 512, 512, 1))

print(y.shape)

with bz2.BZ2File('E:/d/y_hat.pbz2', 'w') as f:
    pickle.dump((x_all, y), f)