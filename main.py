import pandas as pd
from sklearn.datasets import load_diabetes
# from sklearn.datasets import load_wine
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# 2:39 pm — I'm going to play with some datasets from the scikit-learn package
# 3:23 pm — Done experimenting with the diabetes dataset, moving on to the wine dataset
#           to experiment with classification
# 3:56 pm — It turns out that you need to use decision trees instead of a typical
#           classification model... so I should instead use the diabetes dataset again
# 4:45 pm:— Multiple issues: I'm really not good at assigning the variables and plotting the
#           from the datasets. I don't know how to do it. Reusing the code from the tutorial
#           does not work due to the different nature of the dataset.
#           Plus, I wish to get the raw diabetes data instead of the data that has its stdev
#           played with.
#
#           Going to take a break on this and work on pushing this work to the new git branch
#           called 'working'

def main():
    # Load the diabetes dataset
    diabetes = load_diabetes()
    diabetes_df = load_diabetes(as_frame=True)
    X, y = diabetes.data, diabetes.target



    # X = X[['bmi', 's4']]
    # print('diabetes dataset with', X.columns.size, 'columns of categories', X.columns, end='\n')
    # y = pd.Categorical.from_codes(diabetes.data, diabetes.target_names)
    # y = pd.get_dummies(y)
    # print(X.head())
    #
    # print(y)
    # # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    #
    # # Print the dataset to the console
    # sns.scatterplot(
    #     x='bmi',
    #     y='s5',
    #     data=X_test.join(y_test, how='outer')
    # )


if __name__ == "__main__":
    main()
