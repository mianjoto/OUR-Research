import numpy as np


# NONE OF THE CODE IS ORIGINALLY MINE.
# The code below is from following with the video that you linked, found at https://www.youtube.com/watch?v=xG-E--Ak5jg


# 2:07 pm — I understand these 3 different models. I am not sure what the BCI data would look like,
#           but I would have to assume that the K Nearest Neighbors classification method would be best to
#           classify the frequencies of brain waves, keeping in mind the limitations of EEG (it is not
#           possible to classify one's brain waves with 100% accuracy).

def logistic_regression_demo():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    # Step 2: Get data
    x = np.arange(10).reshape(-1, 1)  # Features
    y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])  # Labels
    print(x)
    print(y)

    # Step 3: Create a model and train it
    model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
    model.fit(x, y)

    # Step 4: Evaluate the model
    p_prediction = model.predict_proba(x)
    y_prediction = model.predict(x)  # Returning a 0 or 1
    score_ = model.score(x, y)
    conf_matrix = confusion_matrix(y, y_prediction)
    report = classification_report(y, y_prediction)

    print('x:', x, sep='\n')
    print('y:', y, sep='\n', end='\n\n')

    print('intercept:', model.intercept_)
    print('coef:', model.coef_, end='\n\n')  # Euclidean geometry coefficient

    # Prediction models
    print('y_actual:', y)
    print('  y_pred:', y_prediction)

    # Confusion matrix that compares the y and the y prediction
    print('confusion matrix:', conf_matrix, sep='\n', end='\n\n')
    # FORMAT:   [[TN FN]
    #            [FP TP]]
    # This confusion matrix is useful depending on the desired result,
    # ie, you wouldn't want to reward a machine that says someone doesn't have cancer
    # when they actually do.

    print('report:', report, sep='\n')
    # 80% accuracy!


def k_nearest_neighbors():
    # K Nearest Neighbors is a simple algorithm which stores all available cases
    # and classifies NEW classes based on the similarity measure

    # The K in K Nearest Neighbors is the number of nearest neighbors (if k=3, we are looking for nearest 3 neighbors)
    #   k is usually btwn 3-10
    #   lower k = higher noise
    #   higher k = computationally expensive
    import pandas as pd
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    sns.set()

    breast_cancer = load_breast_cancer()
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    X = X[['mean area', 'mean compactness']]  # Simplify the columns
    y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
    y = pd.get_dummies(y, drop_first=True)  # 1 is benign and 0 is malignant

    print(y)

    # Step 2: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train, y_train)

    # Step 3: Look at X Test data vs Y Test data
    sns.scatterplot(
        x='mean area',
        y='mean compactness',
        hue='benign',
        data=X_test.join(y_test, how='outer')
    )
    # plt.show()

    # Step 4: Predict the data
    y_prediction = knn.predict(X_test)

    plt.scatter(
        X_test['mean area'],
        X_test['mean compactness'],
        c=y_prediction,
        cmap='coolwarm',
        alpha=0.7
    )
    # plt.show()

    print(confusion_matrix(y_test, y_prediction))


def support_vector_machines():
    # Support Vector Machines (SVM)
    # The goal with SVM is to find a hyperplane (collection of vectors) in N dimensional space
    # (where N is the number of features) that distinctly classifies the data points

    # The ultimate goal is to find the plane that has the maximum margin (distance) btwn data points
    # of both or all classes

    # The line that separates the data into two classes is the Support Vector Classifier, or Hard Margin
    #   Hard margin doesn't allow outliers and doesn't allow with non-linearly separable data
    #   We accept soft margins near the hard margins
    #       They sit along the borders of the 2 classes

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_circles
    from sklearn import svm, metrics

    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    samples = 500
    train_prop = 0.8

    # Make data
    x, y = make_circles(n_samples=samples, noise=0.05, random_state=123)

    # Plot data
    df = pd.DataFrame(dict(x=x[:, 0], y=x[:, 1], label=y))

    groups = df.groupby('label')
    fig, ax = plt.subplots()

    ax.margins(0.05)  # Adds padding
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
    ax.legend()
    # plt.show()

    # Min/max ratio
    x = (x - x.min()) / (x.max() - x.min())

    # Linear
    C = 1.0  # SVM regularization parameter
    models = svm.SVC(kernel='rbf', C=C)  # We changed the kernel type from linear to poly to rbf
    models.fit(x, y)

    # Title for the plots
    titles = 'SVC with rbf kernel'

    # Set up 2x2 grid for plotting
    fig, sub = plt.subplots()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = x[:, 0], x[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(sub, models, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    sub.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    sub.set_xlim(-0.25, 1.25)
    sub.set_ylim(-0.25, 1.25)
    sub.set_xlabel('X')
    sub.set_ylabel('Y')
    sub.set_title(titles)

    plt.show()
