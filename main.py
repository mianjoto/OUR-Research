import os
import pickle
from os.path import join
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import int16
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


def main():
    ern_avg, non_ern_avg, dataset, labels, channels = load_eeg_pkl()

    # The channel we will observe is Cn
    chnl, chnl_idx = channels[-1], (len(channels) - 1)

    # Plot signal averages
    errp_labels = ('ErrP Signal Average', 'ms', 'µV')
    non_errp_labels = ('Non-ErrP Signal Average', 'ms', 'µv')

    plot_eeg(ern_avg, errp_labels, clr_pal='bright', fig_num=1,
             columns=channels, show=True)
    plot_eeg(non_ern_avg, non_errp_labels, clr_pal='bright', fig_num=2,
             columns=channels, show=True)

    # Fit data to classifier and plot ROC curve results
    X = pd.DataFrame(dataset[:, chnl_idx, :])
    y = pd.DataFrame(labels)  # 1D nparray of 1s and 0s to indicate ErrPs

    skf_bounds = (2, 5)  # Bounds of how many splits for SKF
    clf = svm.SVC(kernel='rbf', C=5)

    clf, svm_roc, logistic_roc, n_folds = best_n_splits(clf, skf_bounds, X, y)
    plot_roc_curve(chnl, n_folds, skf_bounds, logistic_roc, svm_roc, 3)


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
    return ern_epochs_avg, non_ern_epochs_avg, dataset, labels, channels


def plot_eeg(data: pd.DataFrame, labels: tuple, clr_pal: str = None,
             fig_num: int = None, columns=None, show: bool = None):
    title, xlabel, ylabel = labels

    X = pd.DataFrame(data.T, columns=columns)
    sns.set_style('whitegrid')
    plt.figure(fig_num, dpi=200)
    sns.set_palette(clr_pal)
    sns.lineplot(data=X, dashes=False)
    sns.despine()

    plt.legend(title='Channels', bbox_to_anchor=(1.03, 1), shadow=True,
               facecolor='white', fontsize=8, title_fontsize=12)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show:
        plt.show()


def plot_roc_curve(channel: str, n_folds: int, bounds: tuple,
                   logistic_roc: tuple, svm_roc: tuple,
                   fig_num: int = None, show: bool = True):
    # Make a new matplotlib figure and decorate it
    sns.set_palette('bright')
    plt.figure(fig_num, dpi=200)
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Unpack data
    fpr, tpr, log_auc = logistic_roc
    svm_fpr, svm_tpr, svm_auc = svm_roc

    # Plot ROC data
    plt.plot(svm_fpr, svm_tpr, linestyle='-',
             label=f'{channel} (svm_auc = %.3f)' % svm_auc)
    plt.plot(fpr, tpr, marker='.', label=f'{channel} (auc = %.3f)' %
                                         log_auc)
    plt.legend(title=f'Area under curve for SKF with {n_folds} folds from '
                     f'bounds {str(bounds)}',
               shadow=True, facecolor='white', loc='lower right',
               fontsize=8, title_fontsize=12)
    if show:
        plt.show()


def skf_split_train(n_splits: int, X: pd.DataFrame, y: pd.DataFrame) -> \
                                    Tuple[list, np.ndarray, list, np.ndarray]:
    """Returns a tuple that contains a list of X_train1 and y_train1 objects to
    use to later train a classifier
    """
    skf = StratifiedKFold(n_splits=n_splits)
    X_train_lst = []
    X_test_lst = []
    y_train_lst = []
    y_test_lst = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = np.array(X.iloc[train_index]), \
                          np.array(X.iloc[test_index])
        y_train, y_test = np.array(y.iloc[train_index]).ravel(), np.array(
            y.iloc[test_index]).ravel()

        # Get positive and negative occurrences
        pos = np.where(y_train == 1)[0]
        neg = np.where(y_train == 0)[0]

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

        # Save the values used to train the classifier
        X_train_lst.append(X_train1)
        X_test_lst.append(X_test)
        y_train_lst.append(y_train1)
        y_test_lst.append(y_test)

    # Pick a random X_test and y_test pair to use for testing
    r = np.random.randint(n_splits)
    X_test, y_test = X_test_lst[r], y_test_lst[r]

    return X_train_lst, X_test, y_train_lst, y_test


def svm_fit_clf(clf: svm.SVC, train_data: tuple) -> svm.SVC:
    """Fits data to a passed classifier and returns the classifier. Assumes
    first half of train_data are lists that contain X_train values, and latter
    half are lists that contain y_train values. Both lists are of the same
    length."""
    X_train_lst = []
    y_train_lst = []

    # Unpack X_train and y_train values
    x_y_split = len(train_data[0])
    X_data = train_data[0]
    y_data = train_data[1]

    for i in np.arange(0, x_y_split):
        X_train = X_data[i]
        y_train = y_data[i]
        X_train_lst.append(X_train)
        y_train_lst.append(y_train)

    # Fit the data
    for j in np.arange(x_y_split):
        clf.fit(X_train_lst[j], y_train_lst[j])

    # Return the newly fit classifier
    return clf


def get_roc_data(clf: svm.SVC, X, y) -> Tuple[tuple, tuple]:
    """Return two tuples with logistic ROC data and svm ROC data. Can be
    plotted. Note: does not return the thresholds
    """
    # Use the classifier to predict given the X dataset
    y_pred = clf.predict(X)
    fpr, tpr, threshold = roc_curve(y, y_pred)
    area_under_curve = auc(fpr, tpr)

    y_pred_svm = clf.decision_function(X)
    svm_fpr, svm_tpr, threshold = roc_curve(y, y_pred_svm)
    svm_area_under_curve = auc(svm_fpr, svm_tpr)

    logistic_roc = (fpr, tpr, area_under_curve)
    svm_roc = (svm_fpr, svm_tpr, svm_area_under_curve)

    return logistic_roc, svm_roc


def best_n_splits(clf, fold_bounds: tuple, X, y) -> Tuple[svm.SVC, tuple,
                                                          tuple, int]:
    """Return the classifier that yields the highest tpr after iterating
    through skf folds as many times as bounded by the parameters. Upper bound
    is not exclusive. We test for SVM accuracy as this shows the efficiency
    of the model.
    """
    lower_bound, upper_bound = fold_bounds
    highest_svm_auc = 0.0
    best_log_roc = None
    best_svm_roc = None
    best_n_folds = None
    best_clf = None

    for fold_idx in np.arange(lower_bound, upper_bound):
        # Train and split the data
        X_train_lst, X_test, y_train_lst, y_test = skf_split_train(fold_idx, X,
                                                                   y)
        clf = svm_fit_clf(clf, (X_train_lst, y_train_lst))

        # Get SVM ROC using the entire dataset
        logistic_roc, svm_roc = get_roc_data(clf, X, y)
        svm_fpr, svm_tpr, svm_area_under_curve = svm_roc

        # Store classifier if it yields the highest svm auc
        if svm_area_under_curve > highest_svm_auc:
            highest_svm_auc = svm_area_under_curve
            best_log_roc, best_svm_roc = logistic_roc, svm_roc
            best_n_folds = fold_idx
            best_clf = clf

    print(f'Best number of folds for this trial is {best_n_folds} folds')
    # Return the best classifier and its associated svm auc and logistic auc
    return best_clf, best_svm_roc, best_log_roc, best_n_folds


if __name__ == "__main__":
    main()
