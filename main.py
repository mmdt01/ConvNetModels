"""
Script used to train and test the models: EEGNet, DeepConvNet, ShallowConvNet
Author: Matthys du Toit
Date: 15/04/2024
"""

import numpy as np

# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from tensorflow import keras
from EEGModels import EEGNet
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

##################### Load and epoch data #####################

# define data path
data_path = "subject_4"
file_name = "s4_preprocessed.fif"

# load the preprocessed data
raw = mne.io.read_raw_fif(data_path + "/" + file_name, preload=True)
fs = raw.info.get('sfreq')

# create an event dictionary
event_dict = {
    'motor execution up': 1,
    'motor execution down': 2,
    'visual perception up': 3,
    'visual perception down': 4,
    'imagery up': 5,
    'imagery down': 6,
    'imagery and perception up': 7,
    'imagery and perception down': 8
}

# extract the event information from the raw data
events, event_ids = mne.events_from_annotations(raw, event_id=event_dict)

# extract epochs from the raw data
epochs = mne.Epochs(raw, events, event_id=[3,4,5,6], tmin=0, tmax=3, baseline=None, preload=True)

# shuffle the epochs
permutation = np.random.permutation(len(epochs))
epochs = epochs[permutation]

# plot the epochs
epochs.plot(n_channels=16, scalings={"eeg": 20}, title="Epochs", n_epochs=5, events=True)
plt.show()
print(epochs.drop_log) 

# extract labels from the epochs
labels = epochs.events[:, -1]
print(labels)

# print the shape of the epochs and labels
print("Shape of data: ", epochs.get_data().shape)
print("Shape of labels: ", labels.shape)

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()*1000 # format is in (trials, channels, samples)
y = labels

# print the shape of the epochs and labels
print("Shape of data: ", X.shape)
print("Shape of labels: ", y.shape)

# change this so that it is automatically calculated!
kernels = 1
chans = 64
samples = 751

# take 50/25/25 percent of the data to train/validate/test
X_train = X[0:300,]
Y_train = y[0:300]
X_validate = X[300:450,]
Y_validate = y[300:450]
X_test = X[450:,]
Y_test = y[450:]

############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
Y_train      = np_utils.to_categorical(Y_train-1)
Y_validate   = np_utils.to_categorical(Y_validate-1)
Y_test       = np_utils.to_categorical(Y_test-1)

# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 64 channels and 751 time-points. Set the number of kernels to 1.
X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# compile the model and set the optimizers
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics = ['accuracy']
)

# count number of parameters in the model
numParams    = model.count_params()    

# # set a valid path for your system to record model checkpoints
# checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
#                                save_best_only=True)

# fit the model
fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
                        verbose = 2, validation_data=(X_validate, Y_validate))

# # load optimal weights
# model.load_weights('/tmp/checkpoint.h5')

# make prediction on test set
probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))


