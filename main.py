import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import float64


def main():
    data, channels = load_eeg_pkl()
    plot_eeg(data, channels)


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

    # List that stores the epoch sets, which is the epoch (1 sec) and the number
    # of samples per epoch (200 samples)
    epoch_set_list = []

    # Fill the list an epoch set for each channel
    for i in np.arange(np.size(dataset, axis=1)):
        epoch_set_list.append(dataset[:, i, :])

    # Replace the sets with the epochs that have an associated ErrP value
    ern_occurrences: np.ndarray = np.where(labels == 1)[0]
    for index, epoch_set in enumerate(epoch_set_list):
        epoch_set_list[index]: np.ndarray = epoch_set[ern_occurrences, :]

    # Master list that will hold the list of mean values
    master_epoch_mean_list = []

    # Store mean of each column into a new ndarray
    total_rows = np.shape(epoch_set_list[0])[0]
    for index, epoch_set in enumerate(epoch_set_list):
        column = 0

        # Initialize ndarray to store mean vals for each channel
        mean_ndarray = np.empty(total_rows, dtype=float64)

        # Get mean of each column and store it
        for epoch_row in np.arange(total_rows):
            mean = np.mean(epoch_set[epoch_row, :])
            mean_ndarray[epoch_row] = mean
            column += 1

        # Add ndarray to master list
        master_epoch_mean_list.append(mean_ndarray)

    print('Loaded EEG data')

    if load_all is True:
        return dataset, epoch_set_list, master_epoch_mean_list, labels, \
               event_ids, trial_tags, trial_indexes, dataset_tags, \
               subject_tags, channels

    return master_epoch_mean_list, channels


def plot_eeg(epoch_mean_list: list, channels):
    print('Preparing the EEG data to plot...')

    # Make empty ndarray
    epoch_shape = np.shape(epoch_mean_list[1])
    data = np.empty((epoch_shape[0], 4), dtype=float64)

    # Compile the mean list to one ndarray
    for index, epoch_set in enumerate(epoch_mean_list):
        data[:, index] = epoch_mean_list[index]

    X = pd.DataFrame(data, columns=channels)

    print('Plotting EEG data...')
    sns.lineplot(data=X, dashes=False)
    plt.xlabel('ErrP')
    plt.ylabel('Frequency')
    print('Displayed EEG data')
    plt.show()


if __name__ == "__main__":
    main()
