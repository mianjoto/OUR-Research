import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from practice import best_knn


def breast_cancer_best_knn():
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
    best_knn.plot_best_n_from_range(X_train, X_test_categories[0],
                                    X_test_categories[1],
                                    X_test, y_train, y_test, 1, 100)
