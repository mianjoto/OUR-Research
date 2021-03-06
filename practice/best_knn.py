import csv
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import Bunch

from practice.ndarray_load import combine_tuple_values


def eeg_best_knn():
    sns.set()

    eeg = load_eeg()
    print(f'{eeg.data=}, {len(eeg.data)=}')
    X = pd.DataFrame(eeg.data, columns=eeg.feature_names)
    X_test_categories = ['F4', 'timestamps']
    X = X[X_test_categories]  # Simplify the columns
    y = pd.Categorical.from_codes(eeg.target,
                                  eeg.target_names)
    y = pd.get_dummies(y, drop_first=True)  # 1 is benign and 0 is malignant

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Find the best number of neighbors that yields the least false values
    plot_best_n_from_range(X_train, X_test_categories[0], X_test_categories[1],
                           X_test, y_train, y_test, 1, 100)


def load_eeg(game='obstacle_avoidance', s_num=1, trial_num=1):
    """Return a Bunch that can be used to build a data"""
    if game not in ['obstacle_avoidance', 'binary_goal_search']:
        raise ValueError(f'No such game {game} exists')

    if s_num > 9 or s_num < 0:
        raise ValueError(f'No such subject S{s_num} exists')

    # Instantiate the numpy array
    feature_names = np.array(['labels', 'F4', 'F3', 'Fz', 'Cz',
                              'timestamps'])
    frame = None
    filename = os.getcwd() + f'\\data-v2\\data\\{game}\\observation\\S' \
                             f'{s_num}\\trial-{trial_num}\\filtered\\eeg.csv'
    fdescr = f'EEG data from S{s_num}, trial-{trial_num}'

    data, target = load_data(filename)

    return Bunch(data=data,
                 target=target,
                 frame=frame,
                 DESCR=fdescr,
                 feature_names=feature_names,
                 filename=filename)


def predict_using_test_val(X_train, X_test, y_train, metric,
                           n_neighbors=3):
    # Returns y prediction as ndarray
    return KNeighborsClassifier(n_neighbors, metric=metric).fit(X_train,
                                                                np.ravel(
                                                                    y_train)).predict(
        X_test)


def get_conf_matrix_using_test_val(X_train, X_test, y_train, y_test,
                                   metric, n_neighbors):
    y_pred = predict_using_test_val(X_train, X_test, y_train, metric,
                                    n_neighbors)
    return confusion_matrix(y_pred, y_test)


def get_fn_fp_from_matrix(conf_matrix: confusion_matrix):
    return conf_matrix[0, 1], conf_matrix[1, 0]  # Returns a tuple of the false
    # positives and negatives


def get_fn_fp_using_test_val(X_train, X_test, y_train, y_test,
                             metric, n_neighbors):
    # Create the confusion matrix
    conf_matrix = get_conf_matrix_using_test_val(X_train, X_test,
                                                 y_train,
                                                 y_test, metric,
                                                 n_neighbors)

    # Get the number of false positives and negatives
    fn_fp = get_fn_fp_from_matrix(conf_matrix)
    return fn_fp  # Returns a tuple (FN, FP)


def find_best_n_from_dict(dict_of_knn_results: dict):
    best_n = -1
    lowest_total_false = 999999
    for n_neighbor in dict_of_knn_results:
        conf_matrix = dict_of_knn_results[n_neighbor]
        fn_fp = get_fn_fp_from_matrix(conf_matrix)
        total_false = combine_tuple_values(fn_fp)
        if total_false < lowest_total_false:
            lowest_total_false = total_false
            best_n = n_neighbor
    return best_n


def plot_best_n(X_test, X_test_category1, X_test_category2, y_pred):
    plt.scatter(
        X_test[X_test_category1],
        X_test[X_test_category2],
        c=y_pred,
        cmap='coolwarm',
        alpha=0.7
    )
    plt.show()


def calculate_best_n_from_range(X_train, X_test, y_train, y_test, start=1,
                                stop=10, metric='euclidean'):
    # Handle bad values
    if start <= 0:
        raise ValueError('n must be at least 1')
    if start > stop:
        raise ValueError('Start is higher than the stop')

    # Initialize the memory of the best model
    knn_result = {}

    # Iterate through the models to return the best n_neighbors to use
    for n_neighbors in np.arange(start, stop):
        knn_result[n_neighbors] = get_conf_matrix_using_test_val(
            X_train, X_test,
            y_train,
            y_test, metric,
            n_neighbors)

    # Find the n that yields the least number of FN and FP
    best_n = find_best_n_from_dict(knn_result)
    y_pred = predict_using_test_val(X_train, X_test, y_train, metric, best_n)

    # Get info about the conf matrix to report
    best_conf_matrix = knn_result[best_n]
    fn_fp = get_fn_fp_from_matrix(best_conf_matrix)
    total_false = combine_tuple_values(fn_fp)

    # Print result and return
    print(f'{best_n=} yields {total_false} false values where '
          f'{fn_fp=}\t\tanalyzed with'
          f' {start=}, {stop=}. The best confusion matrix is as '
          f'follows:\n{best_conf_matrix}', end='\n\n')

    return best_n, y_pred


def load_data(filepath):
    """Taken and modified from the sklearn.datasets module. I can't use
    load_data() by itself since it is not init'd
    """
    with open(filepath) as csv_file:
        data_file = csv.reader(csv_file)
        temp: list = next(data_file)
        n_samples = len(temp)
        n_features = sum(1 for _ in csv_file)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)

    return data, target


def plot_best_n_from_range(X_train, X_test_category1, X_test_category2,
                           X_test, y_train,
                           y_test, start=1,
                           stop=10, metric='euclidean'):
    best_n, y_pred = calculate_best_n_from_range(X_train, X_test, y_train,
                                                 y_test,
                                                 start, stop, metric=metric)
    plot_best_n(X_test, X_test_category1, X_test_category2, y_pred, best_n)
