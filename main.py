import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import int16
from sklearn import svm, metrics
from sklearn.metrics import roc_curve, auc, accuracy_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold


def main():
    ern_avg, non_ern_avg, dataset, labels, event_ids, channels = load_eeg_pkl()

    chnl, chnl_idx = channels[-1], (len(channels) - 1)  # The channel we will
    # observe

    # Plot the data
    print('Preparing the EEG data to plot...')
    title = 'ErrP Amplitude Average over 200 Samples'
    xlabel = 'ErrP Data'
    ylabel = 'Amplitude'

    # Assign a DataFrame to the passed datas
    X1 = pd.DataFrame(ern_avg.T, columns=channels)
    X2 = pd.DataFrame(non_ern_avg.T, columns=channels)
    X3 = pd.DataFrame(dataset[:, chnl_idx, :])

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

    # Initialize the classifier
    n_splits = 5
    clf = svm.SVC(kernel='rbf', C=5)
    skf = StratifiedKFold(n_splits=n_splits)
    y = pd.DataFrame(labels)  # 1D nparray of 1s and 0s to indicate ErrPs

    # # Make a new matplotlib figure and decorate it
    plt.figure(4, dpi=100)

    # Decorate the figure
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    print(f'Training and splitting data...')
    X_train_lst = []  # Not used
    y_train_lst = []  # Not used

    for i, (train_index, test_index) in enumerate(skf.split(X3, y)):
        X_train, X_test = np.array(X3.iloc[train_index]), \
                          np.array(X3.iloc[test_index])
        y_train, y_test = np.array(y.iloc[train_index]).ravel(), \
                          np.array(y.iloc[test_index]).ravel()

        # Get positive and negative occurrences
        pos = np.where(y_train == 1)[0]
        neg = np.where(y_train == 0)[0]
        # print(f'{i=}, pos occurrences = {pos.shape[0]} ---- neg occurrences ='
        #       f' {neg.shape[0]}')

        # Get X values that have ERN and do not have ERN
        Xpos = X_train[pos, :]
        Xneg = X_train[neg, :]

        np.vstack((Xpos, Xneg))  # Stack all positive X's on the negative X's

        ds_idx = Xneg.shape[0]  # Get index where we will begin downsampling

        # Sample Xneg array len(XPos) many times
        rand_ds_idx = np.random.randint(0, ds_idx, len(Xpos))

        # Copy the array with the equal pos and equal downsampled neg values
        X_train1 = np.vstack((Xpos, Xneg[rand_ds_idx]))

        ypos = np.ones(Xpos.shape[0])
        yneg = np.zeros(Xpos.shape[0])

        y_train1 = np.concatenate((ypos, yneg), axis=0)

        # Apply weights of positive : neg event to the classifier
        pos_weight = Xpos.shape[0]
        neg_weight = Xneg.shape[0]
        clf.class_weight = {0: neg_weight, 1: pos_weight}

        # Fit the data to the classifier
        clf.fit(X_train1, y_train1)

        # Save the values used to train the classifier
        X_train_lst.append(X_train1)
        y_train_lst.append(y_train1)

    # Predict using the entire dataset
    y_pred = clf.predict(X3)
    fpr, tpr, threshold = roc_curve(y, y_pred)
    area_under_curve = auc(fpr, tpr)

    y_pred_svm = clf.decision_function(X3)
    svm_fpr, svm_tpr, threshold = roc_curve(y, y_pred_svm)
    svm_area_under_curve = auc(svm_fpr, svm_tpr)

    # Plot ROC data
    plt.plot(svm_fpr, svm_tpr, linestyle='-',
             label=f'{chnl} (svm_auc = %.3f)' % svm_area_under_curve)
    plt.plot(fpr, tpr, marker='.', label=f'{chnl} (auc = %.3f)' %
                                         area_under_curve)

    plt.legend(loc='lower right')

    plt.show()


 #y_pred = clf.predict(X_test)

        # # Test accuracy
        # acc = accuracy_score(y_test, y_pred)
        # print('acc = %.2f' % acc)
 # # Convert to np array for compatibility issues
 #        y_train = np.array(y_train).ravel()
 #        y_test = np.array(y_test).ravel()

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


if __name__ == "__main__":
    main()
