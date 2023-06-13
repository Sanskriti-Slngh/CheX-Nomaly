import tensorflow as tf
import gc
import pickle

# import models
import model.unet_model as unet
import random

# variables
train = False
set = "test"
analysis = False
model_name = "unet"
data_name = 'VinBigData'
diseases = ['Aortic Enlargement', 'Atelectasis', 'Calcification' , 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity',
            'Nodule-Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
#diseases = ['Pleural effusion']
all = True
every = False
batch_size = 2
epochs = 1
verbose = 8
callbacks = None
lr=0.00001
augmented = False
special = False
other = True



class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_mae = []
        self.val_mae = []
        self.train_dice = []
        self.val_dice = []
        self.acc_epochs = 0
        super(LossHistory, self).__init__()

    def on_epoch_end(self, epoch, logs):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        gc.collect()

# Load Model and run training
for disease in diseases:
    history = LossHistory()
    if model_name == 'unet':
        model = unet.Model(history, data_name, disease, all, aug=augmented)
    else:
        model = None
        exit("Something went wrong, model not defined")

    model.compile(tf.keras.optimizers.Adam(learning_rate=lr))


    if train and not all:
        print("Getting PIDS")
        if not special:
            model.get_data()
        print("Training model")
        for i in range(10):
            model.train(batch_size, epochs, verbose, callbacks)
            model.save("")
            model.save(i)
    elif train and all:
        for i in range(100):
            model.train_all(batch_size, epochs)
            model.save("")
            model.save(i)

    else:
        f = True
        if not analysis and not other:
            print("giving the evaluation of the model on the " + set)
            model.predict_(batch_size, set)
        if analysis:
            # model training on all data
            if all or special:
                print("getting recall, precsion, f1 score and going trhough each image (hyper analyze each image)")
                model.analyze_all_abnormality(set)

            # go through each specialized model to create final y
            elif every:
                print("lol")
                model.l()
                print("saving for other")
                model.analyze_all(diseases)

            # specific model on any disease
            else:
                while f:
                    f = model.analyze(set)

        # pneumonia dataset
        if other:
           # model.heatmap()
            model.other()
