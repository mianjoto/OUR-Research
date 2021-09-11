import numpy as np
import pandas as pd
# from sklearn.datasets import load_wine
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# 2:39 pm — I'm going to play with some datasets from the scikit-learn
# package
# 3:23 pm — Done experimenting with the diabetes dataset, moving on
# to the wine dataset to experiment with classification
# 3:56 pm — It turns # out that you need to use decision trees instead of a
# typical classification model... so I should instead use the diabetes
# dataset again
# 4:45 pm — # Multiple issues: I'm really not good at assigning the variables
# and # plotting the from the datasets. I don't know how to do it. Reusing the
# code # from the tutorial does not work due to the different nature of the
# dataset. Plus, I wish to get the raw diabetes data instead of the data
# that has its stdev played with.
#
# Going to take a break on this and work on pushing this work to the new git
# branch called 'working'

# 9/11/2021 3:01 pm: Going to try to work on learning the datasets, this time
# with breast cancer information to try and classify as benign or malignant
# 4:04 pm: Giving up on trying to analyze the mean area / smoothness
# and just use the data that was provided in yesterday's practice
# 5:03 pm: I made a function to try to find the n_numbers that yields the
# least number of false negatives/positives. Right now, I am debugging
# due to missing types and mismatching weights
# 5:38 pm: The function find_best_n_neighbors() works and is able to tell the
# best number of neighbors to use for a particular data set using
# KNNClassifier. Now, I am going to make isolate the code inside the for loop
# so that you can find the number of false values for a particular n (which
# will be looped through multiple times)
# 7:52 pm: It's been gruelling, but it's finally done. It can now find the
# best k neighbor value and plot the graph. I don't fully understand
# everything quite yet but I'll learn more with time.

def main():
    sns.set()

    breast_cancer = load_breast_cancer()
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    X_test_categories = ['mean area', 'mean compactness']
    X = X[X_test_categories]  # Simplify the columns
    y = pd.Categorical.from_codes(breast_cancer.target,
                                  breast_cancer.target_names)
    y = pd.get_dummies(y, drop_first=True)  # 1 is benign and 0 is malignant

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Find the best number of neighbors that yields the least false values
    plot_best_n_from_range(X_train, X_test_categories[0], X_test_categories[1],
                           X_test, y_train, y_test, 1, 100)


def combine_tuple_values(t: tuple):
    return sum(list(t))


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


def plot_best_n(X_test, X_test_category1, X_test_category2, y_pred,
                n: int, metric='euclidean'):
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


def plot_best_n_from_range(X_train, X_test_category1, X_test_category2,
                           X_test, y_train,
                           y_test, start=1,
                           stop=10, metric='euclidean'):
    best_n, y_pred = calculate_best_n_from_range(X_train, X_test, y_train,
                                                 y_test,
                                                 start, stop, metric=metric)
    plot_best_n(X_test, X_test_category1, X_test_category2, y_pred, best_n,
                metric=metric)


if __name__ == "__main__":
    main()
