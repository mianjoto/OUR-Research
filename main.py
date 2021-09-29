import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import float64
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def main():
    total_data, errp_data, labels, channels = load_eeg_pkl()
    # plot_eeg(errp_data, channels, 'ErrP', 'Amplitude')
    find_best_accuracy(0.87, errp_data, channels, labels, 1,
                       train_statically=True)
    plot_eeg(total_data, channels, 'epochs', 'amplitude')


def load_eeg_pkl(filepath: str = None, load_all: bool = False):
    """Load EEG data from pkl file

    Loads modified EEG data provided by Ben Poole. The PKL must contain
    labels called `data`, `labels`, `event_ids`, `trail_indexes`, `tags`,
    `dataset`, `subject`, `trial`, and `channels` for this method to work.

    The PKL file is a nested dictionary of dictionaries.

    Parameters
    ----------
    filepath: str, default=None
        filepath for the .pkl file. if not specified, find the file
        ``BGSInt-1-ar.pkl`` in the current working directory

    load_all: bool, default=False
        Whether to return all  dictionaries and tuples

    Returns
    -------
    (epoch_mean_list, channels) : tuple if ``load_all`` is False
        Tuple of data, with the following attributes:

        epoch_mean_list: list
            The list of epochs that are averaged
        channels: list if ``load_all`` is True
            The list that contains the names of the channels

    dataset: ndarray if ``load_all`` is True
        The data that was loaded from the PKL file. Shape varies on the file
    trial_tags: set if ``load_all`` is True
        The set that contains trial tags
    trial_indexes:  dict if ``load_all`` is True
        The dict that contains trial indexes, essentially the same as
        ``trial_tags``
    dataset_tags: set if ``load_all`` is True
        The set that describes the dataset
    subject_tags: set if ``load_all`` is True
        The set that describes the subject identification
    channels: list if ``load_all`` is True
        The list that contains the names of the channels
    """
    if filepath is None:
        d = os.getcwd()
        filepath = join(d, 'BGSInt-1-ar.pkl')
    with open(filepath, 'rb') as file:
        print(f'Loading modified EEG data in directory "{filepath}"')
        file_dataset: dict = pickle.load(file)

        # Unload data into individual dictionaries
        dataset: np.ndarray = file_dataset.get('data')
        labels: np.ndarray = file_dataset.get('labels')
        event_ids = file_dataset.get('event_ids')
        trial_indexes = file_dataset.get('trail_indexes')
        tags = file_dataset.get('tags')
        dataset_tags = tags.get('dataset')
        subject_tags = tags.get('subject')
        trial_tags = tags.get('trial')
        channels = file_dataset.get('channels')

    # Get an array of the indexes of the epochs that have an associated ErrP value
    ern_occurrences: np.ndarray = np.where(labels == 1)[0]

    # Master list that will hold the list of mean values
    epoch_mean_all_channels = []

    # Store mean of each column into a new ndarray
    total_rows = np.shape(dataset)[0]
    print(f'{total_rows=}')
    total_channels = len(channels)

    for channel in np.arange(total_channels):
        # Initialize ndarray to store mean vals for each channel
        mean_ndarray = np.empty(total_rows, dtype=float64)

        # Get mean of each column and store it
        for epoch_row in np.arange(total_rows):
            mean = np.mean(dataset[epoch_row, channel, :])
            mean_ndarray[epoch_row] = mean

        # Add ndarray to master list
        epoch_mean_all_channels.append(mean_ndarray)

    # List that stores the epoch sets, which is the epoch (1 sec) and the number
    # of samples per epoch (200 samples)
    epoch_all_channels = []

    # Fill the list with an epoch set for each channel
    for i in np.arange(np.size(dataset, axis=1)):
        epoch_all_channels.append(dataset[:, i, :])

    print('Loaded EEG data')

    if load_all is True:
        return dataset, epoch_all_channels, epoch_mean_all_channels, labels, \
               event_ids, trial_tags, trial_indexes, dataset_tags, \
               subject_tags, channels
    return epoch_all_channels, epoch_mean_all_channels, labels, channels


def find_best_accuracy(target_acc: float, epoch_all_channels, channels,
                       labels, C_value_bound: int,
                       kernel="linear", train_statically=False, test_size=None):
    """Finds the highest-accuracy models for each channel of the epochs and
    returns a list of y_pred objects representative of each channel to use for
    plotting """
    print('Training svm models')

    channels_dict = {}
    for i in np.arange(4):
        channels_dict.update({i: channels[i]})

    # Store the y prediction for each model
    y_pred_list = []
    index = 0
    # Iterate through each set for each channel
    for epoch_set in epoch_all_channels:
        C_value = C_value_bound
        y_pred = None
        if train_statically:
            # The accuracy will not change if the data is the same every
            # time, regardless of the C value
            acc, y_pred = train_svm(True, epoch_set, labels, kernel, C_value)
        else:
            acc = 0
            # Increase the C value to find the best C if using split data
            while acc < target_acc:
                acc, y_pred = train_svm(False, epoch_set, labels, kernel,
                                        C_value, test_size)
                C_value += 1
        print(f'{acc=}, channel={channels_dict.get(index)}, {C_value=}')

        y_pred_list.append(y_pred)
        index += 1

    print('Trained svm models')
    return y_pred_list


def train_svm(train_statically: bool, epoch_set, labels, kernel: str,
                     C_value: int, test_size: float = None):
    """Statically train the SVM classifier by using the same data each time.
    Should yield the same optimal C_value every single time
    """
    if train_statically:
        x_train = epoch_set[:200].reshape(-1, 1)
        x_test = epoch_set[:200].reshape(-1, 1)
        y_train = np.ravel(labels[:200].reshape(-1, 1))
        y_test = np.ravel(labels[:200].reshape(-1, 1))
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            epoch_set.reshape(-1, 1),
            labels,
            test_size=test_size)

    # Make the classifier
    classifier = svm.SVC(kernel=kernel, C=C_value)
    classifier.fit(x_train, y_train)

    # Predict the value
    y_pred = classifier.predict(x_test)

    # Compare the accuracy of the prediction to an actual variable
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy, y_pred


def plot_eeg(epoch_mean_list: list, channels, xlabel: str, ylabel: str):
    print('Preparing the EEG data to plot...')

    # Make empty ndarray
    epoch_shape = np.shape(epoch_mean_list[1])
    data = np.empty((epoch_shape[0], 4), dtype=float64)

    # Compile the list to one ndarray
    for index, epoch_set in enumerate(epoch_mean_list):
        data[:, index] = epoch_set[:, index]

    X = pd.DataFrame(data, columns=channels)
    print(X)

    print('Plotting EEG data...')
    sns.lineplot(data=X, dashes=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    print('Displayed EEG data')
    plt.show()


if __name__ == "__main__":
    main()
