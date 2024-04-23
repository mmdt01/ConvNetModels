"""
Script used to train and test the models: EEGNet, DeepConvNet, ShallowConvNet: K-FOLD CROSS VALIDATION
Author: Matthys du Toit
Date: 15/04/2024
"""

import mne
from mne import io
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from EEGModels import EEGNet, DeepConvNet, ShallowConvNet
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

##################### Load and epoch data #####################

# define data path
subject = 1
data_path = f"subject_{subject}"
file_name = f"s{subject}_preprocessed.fif"

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

# function for calculating the recall metric
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# function for calculating the precision metric
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# function for calculating the F1-score metric
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# function for loading the raw data and preparing it into labels and epochs
def prepare_data(data_path, file_name, event_dict, class_labels, tmin, tmax):
    # load the preprocessed data
    raw = mne.io.read_raw_fif(data_path + "/" + file_name, preload=True)
    # extract the event information from the raw data
    events, event_ids = mne.events_from_annotations(raw, event_id=event_dict)
    # extract epochs from the raw data
    epochs = mne.Epochs(raw, events, event_id=class_labels, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # plot the epochs (optional)
    epochs.plot(n_channels=16, scalings={"eeg": 20}, title="Epochs", n_epochs=5, events=True)
    print(epochs)
    plt.show()
    # extract and normalize the labels ensuring they start from 1
    labels = epochs.events[:, -1] - min(epochs.events[:, -1]) + 1
    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
    X = epochs.get_data()*1e6 # format is in (trials, channels, samples)
    y = labels
    # print the shape of the epochs and labels
    print('------------------------------------------------------------------------')
    print("Shape of data: ", X.shape)
    print('------------------------------------------------------------------------')
    print("Shape of labels: ", y.shape)
    print('------------------------------------------------------------------------')
    print("Labels: ", y)
    print('------------------------------------------------------------------------')
    # define the number of kernels, channels and samples
    kernels, chans, samples = 1, X.shape[1], X.shape[2]
    # convert data to NHWC (trials, channels, samples, kernels) format. Set the number of kernels to 1.
    X = X.reshape(X.shape[0], chans, samples, kernels)
    # convert labels to one-hot encodings.
    y = np_utils.to_categorical(y-1)
    # return the data
    return X, y, chans, samples

# function for training the model and performing k-fold cross validation
def k_fold_cross_validation(X, y, chans, samples, num_folds, network, task):

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []
    f1_per_fold = []
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1

    for train, test in kfold.split(X, y):
        if network == 'EEGNet':
            # configure the EEGNet-8,2,16 model
            model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples,
                            dropoutRate = 0.5, kernLength = 125, F1 = 8, D = 2, F2 = 16,
                            dropoutType = 'Dropout')
        elif network == 'DeepConvNet':
            # configure the DeepConvNet model
            model = DeepConvNet(nb_classes = 2, Chans = chans, Samples = samples,
                                dropoutRate = 0.5)
        elif network == 'ShallowConvNet':
            # configure the ShallowConvNet model
            model = ShallowConvNet(nb_classes = 2, Chans = chans, Samples = samples,
                                   dropoutRate = 0.5)
        else:
            print('Invalid network type!')
            return
        # compile the model and set the optimizers
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy', f1_m, precision_m, recall_m])
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        # Fit data to model
        history = model.fit(X[train], y[train],
                            batch_size=16,
                            epochs=60,
                            verbose=2)
        # Generate generalization metrics
        scores = model.evaluate(X[test], y[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]}: {scores[0]}; {model.metrics_names[1]}: {scores[1]*100}%; {model.metrics_names[2]}: {scores[2]}; {model.metrics_names[3]}: {scores[3]}; {model.metrics_names[4]}: {scores[4]}')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        f1_per_fold.append(scores[2])
        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}% - F1-Score: {f1_per_fold[i]}')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)}% (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print(f'> F1-Score: {np.mean(f1_per_fold)}')
    print('------------------------------------------------------------------------')
    # write the results to a file in the results folder
    with open(f"results/s{subject}_{network}_results_{task}.txt", "w") as f:
        f.write('------------------------------------------------------------------------\n')
        f.write('Score per fold\n')
        for i in range(0, len(acc_per_fold)):
            f.write('------------------------------------------------------------------------\n')
            f.write(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}% - F1-Score: {f1_per_fold[i]}\n')
        f.write('------------------------------------------------------------------------\n')
        f.write('Average scores for all folds:\n')
        f.write(f'> Accuracy: {np.mean(acc_per_fold)}% (+- {np.std(acc_per_fold)})\n')
        f.write(f'> Loss: {np.mean(loss_per_fold)}\n')
        f.write(f'> F1-Score: {np.mean(f1_per_fold)}\n')
        f.write('------------------------------------------------------------------------\n')

##################### Run the code #####################

num_folds = 10

# epochs=40 gave very slightly better accuracy than 100 epochs for imagery task with DeepConvNet

# Training and test procedure for all experiments per subject:

# 1. Load the preprocessed data and extract epochs for each task:
#       - Manually check epochs and remove any bad epochs
#       - (Everything from now onwards is automated...)
# 2. For each classification task, train the model using k-fold cross validation and evaluate performance with:
#       - DeepConvNet
#       - ShallowConvNet
#       - EEGNet
# 3. Save the results to a file in the results folder
# 4. Repeat for each subject

# 1. Load the preprocessed data and extract epochs for each task:
X_1, y_1, chans_1, samples_1 = prepare_data(data_path, file_name, event_dict, class_labels=[1,2], tmin=0, tmax=3)
X_2, y_2, chans_2, samples_2 = prepare_data(data_path, file_name, event_dict, class_labels=[3,4], tmin=0, tmax=3)
X_3, y_3, chans_3, samples_3 = prepare_data(data_path, file_name, event_dict, class_labels=[5,6], tmin=0, tmax=3)
X_4, y_4, chans_4, samples_4 = prepare_data(data_path, file_name, event_dict, class_labels=[7,8], tmin=0, tmax=3)

# 2. For each classification task, train the model using k-fold cross validation and evaluate performance with:
k_fold_cross_validation(X_1, y_1, chans_1, samples_1, num_folds, 'DeepConvNet', 'motor-execution')
k_fold_cross_validation(X_2, y_2, chans_2, samples_2, num_folds, 'DeepConvNet', 'perception')
k_fold_cross_validation(X_3, y_3, chans_3, samples_3, num_folds, 'DeepConvNet', 'imagery')
k_fold_cross_validation(X_4, y_4, chans_4, samples_4, num_folds, 'DeepConvNet', 'imagery-perception')

k_fold_cross_validation(X_1, y_1, chans_1, samples_1, num_folds, 'ShallowConvNet', 'motor-execution')
k_fold_cross_validation(X_2, y_2, chans_2, samples_2, num_folds, 'ShallowConvNet', 'perception')
k_fold_cross_validation(X_3, y_3, chans_3, samples_3, num_folds, 'ShallowConvNet', 'imagery')
k_fold_cross_validation(X_4, y_4, chans_4, samples_4, num_folds, 'ShallowConvNet', 'imagery-perception')

k_fold_cross_validation(X_1, y_1, chans_1, samples_1, num_folds, 'EEGNet', 'motor-execution')
k_fold_cross_validation(X_2, y_2, chans_2, samples_2, num_folds, 'EEGNet', 'perception')
k_fold_cross_validation(X_3, y_3, chans_3, samples_3, num_folds, 'EEGNet', 'imagery')
k_fold_cross_validation(X_4, y_4, chans_4, samples_4, num_folds, 'EEGNet', 'imagery-perception')



