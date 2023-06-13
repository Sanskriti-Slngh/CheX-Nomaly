import pickle
import tensorflow as tf
from lungabnormality.code.model.DataGen import CustomDataGen
from lungabnormality.code.model.datagenall import CustomDataGenAll
from lungabnormality.code.model.datagenspecial import CustomDataGenSpecial
from tensorflow.keras.models import load_model
import os
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Reduction
import pydicom
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import bz2
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras



class BaseModel:
    """
    data_name: numerous datasets require different methods of processing and have different folders; what dataset is being trained/validated/tested
               VinBigData, RSNA, JSRT, CANDID-PTX
    disease: only usefull if "all" is False meaning seperate datasets
    all: True when training/validating/testing supermodel
    """

    def __init__(self, data_name, disease, aug, all=False):

        self.disease = disease
        self.augment = aug
        self.special = False

        if False:
            self.main_directory = 'DNE'
            self.image_data_csv = 'DNE'

        else:
            if data_name == 'VinBigData':
                self.main_directory = 'F:/lungabnormality/data/'
                self.image_data_csv = 'D:/tiya2023/lungabnormality/data/train.csv/train.csv'

                if self.disease == 'Nodule/Mass':
                    self.disease = "Nodule-Mass"
                self.PID_filename = "D:/tiya2023/lungabnormality/data/diseases/" + self.disease + ".dat"

            else:
                print("DATASET NOT FOUND")
                exit()

        # name where I store and load trained models in
        self.name = "D:/tiya2023/lungabnormality/model/" + self.disease

        if all:
            self.name = "D:/tiya2023/lungabnormality/model/all"

        if self.special:
            self.name = "D:/tiya2023/lungabnormality/model/special"

        # load model if model is already there
        print(self.name + '.h5')
        if os.path.isfile(self.name + '.h5'):
            print("Loading model from " + self.name + '.h5')
            self.model = load_model(self.name + '.h5', custom_objects={'dice_coef' : self.dice_coef, 'focal_loss': self.focal_loss, 'dice_loss': self.dice_loss})

    # load model
    def model_load_again(self, i):
        print("Loading model from " + self.name + str(i) + '.h5')
        self.model = load_model(self.name + str(i) + '.h5', custom_objects={'dice_coef' : self.dice_coef, 'focal_loss': self.focal_loss, 'dice_loss': self.dice_loss})

    # get the PIDS
    def get_data(self):
        file = open(self.PID_filename, 'rb')
        self.train_PIDs, self.val_PIDs, self.test_PIDs = pickle.load(file)
        file.close()

    def dice_coef(self, targets, inputs, smooth=10):
        y_true_f = K.flatten(targets)
        y_pred_f = K.flatten(inputs)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice

    # compile model
    def compile(self, optimizer):
        self.model.compile(optimizer=optimizer,
                           loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25,gamma=2),
                           metrics=[tf.keras.metrics.MeanAbsoluteError()])

    def dice_loss(self, targets, inputs):
        return 1 - self.dice_coef(targets, inputs)

    def focal_loss(self, y_true, y_pred):
        alpha = 0.25
        gamma = 2
        y_true_f = K.flatten(y_true[:, :, :, 0])
        y_pred_f = K.flatten(y_pred[:, :, :, 0])
        focal = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.NONE,
                                                    gamma=gamma,
                                                    alpha=alpha)(y_true_f, y_pred_f)
        l = K.sum(focal) / (512*512)

        return l

    # save model every i epochs
    def save(self, i):
        print("Saving model into " + self.name + str(i))
        self.model.save(self.name + str(i) + ".h5")
        with open(self.name + str(i) + ".aux_data", "wb") as fout:
            pickle.dump((self.history.train_losses, self.history.val_losses), fout)

    # train binary segmentation models
    def train(self, batch_size, epochs, verbose, callbacks):
        mc = ModelCheckpoint(self.name + '.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        if self.special:
            print("Training special model")
            traingen = CustomDataGenSpecial(data_set='train', aug=self.augment, create=True)
            valgen = CustomDataGenSpecial(data_set='val', aug=False, create=True)
        else:
            print("Train normal binary segmentation model for " + str(self.disease))
            traingen = CustomDataGen(self.train_PIDs, self.image_data_csv, self.main_directory, self.disease,
                                     batch_size=batch_size, data_set='train', aug=self.augment, create=True)
            valgen = CustomDataGen(self.val_PIDs, self.image_data_csv, self.main_directory, self.disease,
                                   batch_size=batch_size, data_set='val', aug=False, create=True)

        h = self.model.fit(traingen, batch_size=batch_size, epochs=epochs, validation_data=valgen, callbacks=[self.history])

        pd.DataFrame(h.history).plot(figsize=(8, 5))
        plt.show()

    # train on all data
    def train_all(self, batch_size, epochs):
        traingen = CustomDataGenAll(batch_size=batch_size, data_set='train')

        valgen = CustomDataGenAll(batch_size=batch_size, data_set='val')

        h = self.model.fit(traingen, epochs=epochs, validation_data=valgen, callbacks=self.history)

        # pd.DataFrame(h.history).plot(figsize=(8, 5))
        # plt.show()

    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, :, :, :]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def heatmap(self):
        print(self.model.summary())
        last_conv_layer_name = 'conv2d_15'
        is_file = 'D:/tiya2023/lungabnormality/data/allvinbig' + 'val' + '.pbz2'
        data = bz2.BZ2File(is_file)
        self.x, self.y = pickle.load(data)

        for i in range(len(self.x)):
            img_array = self.x[i]
            y = self.y[i]
            img_array = np.array(img_array)
            y = np.array(y)
            img_array = np.reshape(img_array, (1, 512, 512, 1))
            y = np.reshape(y, (512, 512))
            pred = self.model.predict(img_array)
            # Remove last layer's softmax
            self.model.layers[-1].activation = None

            # Print what the top predicted class is
            decode_predictions = keras.applications.xception.decode_predictions
            preds = self.model.predict(img_array)
            # print("Predicted:", decode_predictions(preds, top=1)[0])

            # Generate class activation heatmap
            heatmap = self.make_gradcam_heatmap(img_array, self.model, last_conv_layer_name)

            # Display heatmap
            j, k = plt.subplots(1, 3, figsize=(20, 20))
            k[0].imshow(np.reshape(img_array, (512, 512)))
            k[0].contour(y)
            k[1].matshow(heatmap)
            k[2].imshow(np.reshape(img_array, (512, 512)))
            k[2].contour((np.reshape(pred, (512, 512)))>0.5)

            plt.show()

    # plan 1
    # ensembling of models directly
    def ensemble_metrics(self, x, y, diseases):
        # normalize x
        y_pred = []
        for i in x:
            min_value = np.min(i)
            max_value = np.max(i)
            range = max_value - min_value
            i = (i - min_value) / range
            y_pred.append(self.combined_pred(diseases, np.reshape(i, (1, 512, 512, 1))))

        y_pred = np.reshape(y_pred, (len(y), 512, 512, 1))
        y = np.reshape(y, (len(y), 512, 512, 1))
        return self.my_mae(y_pred, y)

    # get basic metrics for binary models
    def predict_(self, batch_size, set):
        print(self.disease)
        import bz2
        data = bz2.BZ2File('D:/tiya2023/lungabnormality/data/allvinbig' + set + '.pbz2')
        d = 'D:/tiya2023/lungabnormality/data/allvinbig' + set + '.pbz2'
        self.x, self.y = pickle.load(data)
        x = np.array(self.x)
        y = np.array(self.y)

        x = np.reshape(x, (x.shape[0], 512, 512, 1))
        y = np.reshape(y, (x.shape[0], 512, 512, 1))

        m = tf.keras.metrics.MeanIoU(num_classes=2)

        print("here")
        pred = self.model.predict(x)
        mae = m(y_true=y, y_pred=pred)
        print(mae)

        return self.model.evaluate(x, y, batch_size)

    # the y_prediction from every binary model
    def combined_pred(self, diseases, x):
        preds = []
        for d in diseases:
            n = "D:/tiya2023/lungabnormality/model/" + d
            m = load_model(n + '.h5', custom_objects={'dice_coef' : self.dice_coef, 'focal_loss': self.focal_loss})
            preds.append(m.predict(x))

        y_hat = np.zeros((1, 512, 512, 1))
        for i in preds:
            y_hat = np.add(y_hat, i)

        return y_hat

    # analyze every binary models segmentation
    def analyze_all(self, diseases):
        is_file = 'D:/tiya2023/lungabnormality/data/allvinbigtrain.pbz2'
        data = bz2.BZ2File(is_file)
        self.x, self.y = pickle.load(data)
        print("got data")
        print(len(self.y))

        self.x = self.x[9000:]
        self.y = self.y[9000:]

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.x = np.reshape(self.x, (self.y.shape[0], 512, 512, 1))
        self.y = np.reshape(self.y, (self.y.shape[0], 512, 512, 1))

        preds = []

        for d in diseases:
            print(d)
            n = "D:/tiya2023/lungabnormality/model/" + d
            m = load_model(n + '.h5', custom_objects={'dice_coef': self.dice_coef, 'focal_loss': self.focal_loss})
            a = m.predict(self.x, batch_size=8)
            with bz2.BZ2File('D:/tiya2023/lungabnormality/data/y_hat' + d + '2.pbz2', 'w') as f:
                pickle.dump((a), f)
            preds.append(a)

        preds = np.array(preds)
        print(preds.shape)

        pred = (np.max(preds, axis=0))

        print(self.y.shape)
        print(pred.shape)
        print(pred)

        print("low")
        f = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2, reduction=Reduction.SUM_OVER_BATCH_SIZE)
        m = tf.keras.losses.MeanAbsoluteError()
        print("here")
        mae = m(y_true=self.y, y_pred=pred)
        print(mae)

        loss = f(y_true=self.y, y_pred=pred)
        print(loss)

    # analayze binary model semgentation on speicifc pid
    def analyze(self, set):
        is_file = 'D:/tiya2023/lungabnormality/data/' + self.disease + set + '.pbz2'
        data = bz2.BZ2File(is_file)
        self.x, self.y = pickle.load(data)
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for ind, img in enumerate(self.x):
            img = np.array(img)
            img = np.reshape(img, (1, 512, 512, 1))
            y = np.reshape(np.array(self.y[ind]), (1, 512, 512, 1))

            pred = self.model.predict(img)
            # print(np.sum(pred > 0.5), np.sum(y))

            a = np.sum(pred > 0.5)
            b = np.sum(y)
            print(a, b)
            if a > 0 and b > 0:
                tp += 1
                img = np.reshape(img, (512, 512))
                y = np.reshape(y, (512, 512))
                pred = np.reshape(pred, (512, 512))
                self.prediction_plot([img], [y], [pred])
                plt.imshow(img)
                plt.show()
                #return False
            elif a > 0 and b == 0:
                fp += 1
            elif a == 0 and b > 0:
                fn += 1
            else:
                tn += 1

            #img = np.reshape(img, (512, 512))
            #y = np.reshape(y, (512, 512))
            #pred = np.reshape(pred, (512, 512))
            #self.prediction_plot([img], [y], [pred])

        print(fp, fn, tp, tn)
        try:
            print("Recall is " + str(tp/(tp+fn)))
        except:
            print("0")
        try:
            print("Precision is " + str(tp/(tp+fp)))
        except:
            print("0")
        try:
            print("F1 score is " + str((tp)/(tp+1/2*(fp+fn))))
        except:
            print("0")

    def l(self):
        is_file = 'D:/tiya2023/lungabnormality/data/allvinbig' + 'val' + '.pbz2'
        data = bz2.BZ2File(is_file)
        self.x, self.y = pickle.load(data)

        for ind, img in enumerate(self.x):
            img = np.array(img)
            img = np.reshape(img, (1, 512, 512, 1))
            y = np.reshape(np.array(self.y[ind]), (1, 512, 512, 1))


            b = np.sum(y)
            if b > 0:
                img_1 = np.reshape(img, (512, 512))
                y = np.reshape(y, (512, 512))

                preds = []
                for d in ['Aortic Enlargement', 'Atelectasis', 'Calcification' , 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule-Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']:
                    print(d)
                    n = "D:/tiya2023/lungabnormality/model/" + d
                    m = load_model(n + '.h5',
                                   custom_objects={'dice_coef': self.dice_coef, 'focal_loss': self.focal_loss})
                    a = m.predict(img)

                    a = np.reshape(a, (512, 512))
                    preds.append(a)

                self.prediction_plot([img_1], [y], preds)

    def analyze_all_abnormality(self, set):
        is_file = 'D:/tiya2023/lungabnormality/data/allvinbig' + set + '.pbz2'
        # print(self.disease)
        # is_file = 'D:/tiya2023/lungabnormality/data/' + self.disease + set + '.pbz2'
        data = bz2.BZ2File(is_file)
        self.x, self.y = pickle.load(data)

        if False:
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
                image = (image / (max_value - min_value)).astype(np.float16)

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

                        bb = [int(x_min * self.size / height), int(y_min * self.size / width),
                              int(x_max * self.size / height),
                              int(y_max * self.size / width)]
                        mask[bb[1]:bb[3], bb[0]:bb[2], :] = 1

                self.x.append(image)
                self.y.append(mask)

        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for ind, img in enumerate(self.x):
            img = np.array(img)
            img = np.reshape(img, (1, 512, 512, 1))
            y = np.reshape(np.array(self.y[ind]), (1, 512, 512, 1))

            pred = self.model.predict(img)

            a = np.sum(pred > 0.5)
            b = np.sum(y)
            if a > 0 and b > 0:
                tp += 1
            elif a > 0 and b == 0:
                fp += 1
            elif a == 0 and b > 0:
                fn += 1
            else:
                tn += 1

            img = np.reshape(img, (512, 512))
            y = np.reshape(y, (512, 512))
            pred = np.reshape(pred, (512, 512))
            self.prediction_plot([img], [y], [pred])

        print(fp, fn, tp, tn)
        try:
            print("Recall is " + str(tp/(tp+fn)))
        except:
            print("0")
        try:
            print("Precision is " + str(tp/(tp+fp)))
        except:
            print("0")
        try:
            print("F1 score is " + str((tp)/(tp+1/2*(fp+fn))))
        except:
            print("0")

        f = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.5, gamma=2, reduction=Reduction.SUM_OVER_BATCH_SIZE)
        m = tf.keras.losses.MeanAbsoluteError()
        print("here")
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.x = np.reshape(self.x, (self.y.shape[0], 512, 512, 1))
        self.y = np.reshape(self.y, (self.y.shape[0], 512, 512, 1))
        a = self.model.predict(self.x)
        mae = m(y_true=self.y, y_pred=a)
        print(mae)

        loss = f(y_true=self.y, y_pred=a)
        print(loss)

        self.model.evaluate(self.x, self.y)
        exit()

    def other(self):
        x = []
        y = []
        c = 0
        d = 1
        pids = []
        directory = 'D:/Manasvini-2022/sf2022/data/stage_2_train_images/'
        for filename in os.listdir(directory):
            pids.append(filename.split('.dcm')[0])
        for qw, pid in enumerate(pids):
            if qw <= 50:
                f = 'D:/Manasvini-2022/sf2022/data/stage_2_train_images/' + pid + '.dcm'
                ds = pydicom.dcmread(f).pixel_array
                train_df = pd.read_csv('D:/Manasvini-2022/sf2022/data/stage_2_train_labels.csv')
                class_rows = train_df.loc[train_df['patientId'] == pid]
                mask = np.zeros((512, 512))
                for i in range(len(class_rows)):
                    x_min = class_rows.iloc[i]['x']
                    y_min = class_rows.iloc[i]['y']
                    x_max = x_min + class_rows.iloc[i]['width']
                    y_max = y_min + class_rows.iloc[i]['height']
                    #print(x_min, x_max, y_min, y_max)
                    if np.isnan(x_max):
                        c += 1
                        continue
                    else:
                        d += 1
                        mask[int(y_min*512/1024):int(y_max*512/1024), int(x_min*512/1024):int(x_max*512/1024)] = 1

                ds = resize(ds, (512, 512), preserve_range=True)
                #ds = np.reshape(ds, (1, 512, 512, 1))
                #mask = np.reshape(mask, (1, 512, 512, 1))
                x.append(ds)
                y.append(mask)

        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for ind, img in enumerate(x):
            img = np.array(img)
            img = np.reshape(img, (1, 512, 512, 1))
            y_l = np.reshape(y[ind], (1, 512, 512, 1))

            pred = self.model.predict(img)

            q = np.reshape(y_l, (1, 512, 512, 1))
            w = np.reshape(pred, (1, 512, 512, 1))

            m = tf.keras.metrics.MeanIoU(num_classes=2)
            p = m(q, w)

            a = p > 0.3
            b = np.sum(y_l)
            # print(a, b)
            if a and b > 0:
                tp += 1
            elif a and b == 0:
                fp += 1
            elif not a and b > 0:
                fn += 1
            else:
                tn += 1

            img1 = np.reshape(img, (512, 512))
            y1 = np.reshape(y_l, (512, 512))
            pred1 = np.reshape(pred, (512, 512))
            self.prediction_plot([img1], [y1], [pred1])

        print(fp, fn, tp, tn)
        print(tp / (tp + fn + fp))
        try:
            print("Recall is " + str(tp / (tp + fn)))
        except:
            print("0")
        try:
            print("Precision is " + str(tp / (tp + fp)))
        except:
            print("0")
        try:
            print("F1 score is " + str((tp) / (tp + 1 / 2 * (fp + fn))))
        except:
            print("0")

        x = np.reshape(x, (len(x), 512, 512, 1))
        y = np.reshape(y, (len(y), 512, 512, 1))

        m = tf.keras.losses.MeanAbsoluteError()
        print("here")
        pred = self.model.predict(x)
        mae = m(y_true=y, y_pred=pred)

        self.model.evaluate(x, y, 8)
        print(c,d)


    # get the plot of the prediciton
    def prediction_plot(self, x, y, pred):
        j, k = plt.subplots(1, 3, figsize=(20, 20))
        k[0].imshow(x[0])
        k[0].contour(y[0])
        k[1].imshow(y[0])
        k[2].imshow(x[0])
        for i in pred:
            i = i > 0.5
            k[2].contour(i)

        plt.show()


