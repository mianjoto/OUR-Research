import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import int16
from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def main():
    ern_avg, non_ern_avg, dataset, labels, event_ids, channels = load_eeg_pkl()

    # Plot the data
    print('Preparing the EEG data to plot...')
    title = 'ErrP Amplitude Average over 200 Samples'
    xlabel = 'ErrP Data'
    ylabel = 'Amplitude'

    # Assign a DataFrame to the passed datas
    X1 = pd.DataFrame(ern_avg.T, columns=channels)
    X2 = pd.DataFrame(non_ern_avg.T, columns=channels)
    X3 = pd.DataFrame(dataset[:, :, 1], columns=channels)

    # ErrP data
    plt.figure(1)
    sns.lineplot(data=X1, dashes=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Non-ErrP data
    plt.figure(2)
    sns.lineplot(data=X2, dashes=False)
    plt.title(f'Non-{title}')
    plt.xlabel(f'Non-{xlabel}')
    plt.ylabel(ylabel)

    # All data
    plt.figure(3)
    sns.lineplot(data=X3, dashes=False)
    plt.title(f'Total {title}')
    plt.xlabel(f'All {xlabel}')
    plt.ylabel(ylabel)

    # plt.show()

    # Train the classifier
    accuracy_list = []
    classifier = svm.SVC(kernel='rbf', C=4)
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    y = pd.DataFrame(labels)  # 1D nparray of 1s and 0s to indicate ErrPs

    for train_index, test_index in skf.split(X3, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X3.iloc[train_index], X3.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_train)
        acc = accuracy_score(y_pred, y_test)
        accuracy_list.append(acc)

    print(f'{accuracy_list=}')
    avg_acc = np.mean(accuracy_list)
    print(f'{avg_acc=}')


def load_eeg_pkl(filepath: str = None, return_all: bool = False):
    """Load EEG data from pkl file

    Loads modified EEG data provided by Ben Poole. The PKL must contain
    labels called `data`, `labels`, `event_ids`, `trail_indexes`, `tags`,
    `dataset`, `subject`, `trial`, and `channels` for this method to work.

    The PKL file is a nested dictionary of dictionaries.
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

    # Get ndarray of indexes of epochs that have associated ERN value
    ern_occurrences: np.ndarray = np.where(labels == 1)[0]
    non_ern_occurrences: np.ndarray = np.where(labels == 0)[0]

    # Store a labels df from a 2D, (1013, 2) ndarray
    total_rows = ern_occurrences.shape[0] + non_ern_occurrences.shape[0]
    labels_array = np.zeros((total_rows, 2), dtype=int16)
    for row in np.arange(total_rows):
        if row in ern_occurrences:
            labels_array[row, 1] = 1
        else:
            labels_array[row, 0] = 1

    labels_df: pd.DataFrame = pd.DataFrame(labels_array, columns=event_ids)

    # Get array of the epochs that DO and DO NOT have an associated ErrP value
    ern_epochs = dataset[ern_occurrences]
    non_ern_epochs = dataset[non_ern_occurrences]

    # Get average across all samples that have and do not have associated ERN
    # plus original dataset
    ern_epochs_avg = np.mean(ern_epochs, axis=0)
    non_ern_epochs_avg = np.mean(non_ern_epochs, axis=0)

    if return_all:
        return dataset, ern_epochs, ern_epochs_avg, non_ern_epochs, \
               non_ern_epochs_avg, labels, labels_df, event_ids, trial_tags, \
               trial_indexes, \
               dataset_tags, subject_tags, channels
    return ern_epochs_avg, non_ern_epochs_avg, dataset, labels, event_ids, channels


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


if __name__ == "__main__":
    main()
