"""
Script used to train and test the models: EEGNet, DeepConvNet, ShallowConvNet: K-FOLD CROSS VALIDATION
Author: Matthys du Toit
Date: 15/04/2024
"""

import mne
from mne import io
from sklearn.model_selection import KFold
import numpy as np

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
epochs = mne.Epochs(raw, events, event_id=[5,6], tmin=0, tmax=3, baseline=None, preload=True)
# shuffle the epochs
permutation = np.random.permutation(len(epochs))
epochs = epochs[permutation]

# # plot the epochs
# epochs.plot(n_channels=16, scalings={"eeg": 20}, title="Epochs", n_epochs=5, events=True)
# print(epochs)
# print(len(epochs))
# plt.show()
# print(epochs.drop_log) 

# extract and normalize the labels ensuring they start from 1
labels = epochs.events[:, -1] - min(epochs.events[:, -1]) + 1
print(labels)

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()*1000 # format is in (trials, channels, samples)
y = labels

# print the shape of the epochs and labels
print("Shape of data: ", X.shape)
print("Number of samples: ", X.shape[2])
print("Shape of labels: ", y.shape)

# change this so that it is automatically calculated!
kernels = 1
chans = X.shape[1] # 64
samples = X.shape[2] # 751

# convert labels to one-hot encodings.
y = np_utils.to_categorical(y-1)

# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 64 channels and 751 time-points. Set the number of kernels to 1.
X = X.reshape(X.shape[0], chans, samples, kernels)

############################# K-FOLD CROSS VALIDATION ##################################

num_folds = 5

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X, y):

    # configure the EEGNet-8,2,16 model
    model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples,
                     dropoutRate = 0.5, kernLength = 125, F1 = 8, D = 2, F2 = 16,
                     dropoutType = 'Dropout')
    
    # compile the model and set the optimizers
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics = ['accuracy']
    )

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(X[train], y[train],
                        batch_size=16,
                        epochs=100,
                        verbose=2)
    
    # Generate generalization metrics
    scores = model.evaluate(X[test], y[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')





############################# EEGNet portion ##################################

# # convert labels to one-hot encodings.
# Y_train      = np_utils.to_categorical(Y_train-1)
# Y_validate   = np_utils.to_categorical(Y_validate-1)
# Y_test       = np_utils.to_categorical(Y_test-1)

# # convert data to NHWC (trials, channels, samples, kernels) format. Data 
# # contains 64 channels and 751 time-points. Set the number of kernels to 1.
# X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
# X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
# X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# # model configurations may do better, but this is a good starting point)
# model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
#                dropoutRate = 0.5, kernLength = 125, F1 = 8, D = 2, F2 = 16, 
#                dropoutType = 'Dropout')

# # compile the model and set the optimizers
# model.compile(
#     loss='binary_crossentropy', 
#     optimizer='adam', 
#     metrics = ['accuracy']
# )

# # count number of parameters in the model
# numParams    = model.count_params()    

# # set a valid path for your system to record model checkpoints
# checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
#                                save_best_only=True)

# # fit the model
# fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
#                         verbose = 2, validation_data=(X_validate, Y_validate),
#                         callbacks=[checkpointer])

# # load optimal weights
# model.load_weights('/tmp/checkpoint.h5')

# # make prediction on test set
# probs       = model.predict(X_test)
# preds       = probs.argmax(axis = -1)  
# acc         = np.mean(preds == Y_test.argmax(axis=-1))
# print("Classification accuracy: %f " % (acc))