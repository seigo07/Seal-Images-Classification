import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


# Loading the binary data
def load_data():
    # print('Loading multi-class data...')
    X_train_df = pd.read_csv("multi/X_train.csv", header=None)
    # X_train_df.info()

    y_train_df = pd.read_csv("multi/Y_train.csv", header=None)
    # print('Unique values', y_train_df.iloc[:, 0].unique())

    X_test_df = pd.read_csv("multi/X_test.csv", header=None)
    # X_test_df.info()

    data_dict = {
        'X_train': X_train_df,
        'X_test': X_test_df,
        'y_train': y_train_df
    }
    # print('\n')

    return data_dict


# Cleaning the data by removing the features taken from a random normal distribution
def clean_data(data_dict):
    print('Cleaning data...')
    X_train_df = data_dict['X_train']
    X_test_df = data_dict['X_test']
    y_train_df = data_dict['y_train']

    X_train_df = X_train_df.drop(X_train_df.columns[900:916], axis=1)  # df.columns is zero-based pd.Index
    X_test_df = X_test_df.drop(X_test_df.columns[900:916], axis=1)  # df.columns is zero-based pd.Index

    print('X train after cleaning:')
    X_train_df.info()
    print('X_test after cleaning')
    X_test_df.info()
    data_dict = {
        'X_train': X_train_df,
        'X_test': X_test_df,
        'y_train': y_train_df
    }
    print('\n')
    return data_dict


# Visualising the target frequencies to see whether the dataset is imbalanced.
def plot_target_frequency(data_dict):
    y_train = data_dict['y_train']
    y_train.columns = ['label']
    print(y_train.columns)
    total = float(len(y_train))
    print('total', total)
    plot = sns.countplot(x='label', data=y_train)
    for p in plot.patches:
        height = p.get_height()
        plot.text(p.get_x() + p.get_width() / 2.,
                  height + 3,
                  '{:1.3f}'.format(height / total),
                  ha="center")
    plt.show()


# Defining the estimator as well as the grid search for training KNN.
# The estimator is a pipeline in which the features are scaled, PCA is performed and a KNN classifier is trained.
def kn_cross_validate_pca(X_train, y_train, scorer):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline(
        [('sc', StandardScaler()), ('pca', PCA(n_components=0.99, svd_solver='full')), ('cf', KNeighborsClassifier())])

    params = {
        'cf__n_neighbors': [3],
        'cf__leaf_size': [30],
        'cf__p': [2]
    }

    print('Grid: ', params)

    print('Scorer: ', scorer)

    cf = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring=scorer, refit=scorer[0])

    start = time.time()
    cf.fit(X_train, y_train)
    end = time.time()
    print('K-nearest neighbors cross-val time elapsed: ', end - start)

    print('Best params: ', cf.best_params_)

    print('PCA number of components', cf.best_estimator_.named_steps['pca'].n_components_)

    balanced_acc_score = cf.best_score_ * 100

    acc_score = cf.cv_results_['mean_test_accuracy'][cf.best_index_] * 100

    print("Best cross-val balanced accuracy score: " + str(round(balanced_acc_score, 2)) + '%')
    print("Best cross-val accuracy score: " + str(round(acc_score, 2)) + '%')

    cv_results = pd.DataFrame(cf.cv_results_)
    # display(cv_results)

    print('\n')
    return cf.best_estimator_


# Defining the estimator as well as the grid search for training Random Forest Classifier.
# The estimator is a pipeline in which the features are scaled,
# PCA is performed and a Random Forest Classifier classifier is trained.
def rf_cross_validate_pca(X_train, y_train, scorer):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline(
        [('sc', StandardScaler()), ('pca', PCA(n_components=0.99, svd_solver='full')),
         ('cf', RandomForestClassifier())])

    params = {
        'cf__n_estimators': [400],
        'cf__max_depth': [None],
        'cf__min_samples_split': [2],
    }

    print('Grid: ', params)

    print('Scorer: ', scorer)

    cf = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring=scorer, refit=scorer[0])

    start = time.time()
    cf.fit(X_train, y_train)
    end = time.time()
    print('Random Forest cross-val time elapsed: ', end - start)

    print('Best params: ', cf.best_params_)

    print('PCA number of components', cf.best_estimator_.named_steps['pca'].n_components_)

    balanced_acc_score = cf.best_score_ * 100

    acc_score = cf.cv_results_['mean_test_accuracy'][cf.best_index_] * 100

    print("Best cross-val balanced accuracy score: " + str(round(balanced_acc_score, 2)) + '%')
    print("Best cross-val accuracy score: " + str(round(acc_score, 2)) + '%')

    cv_results = pd.DataFrame(cf.cv_results_)
    # display(cv_results)

    print('\n')
    return cf.best_estimator_


def svc_cross_validate_pca(X_train, y_train, scorer):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline(
        [('sc', StandardScaler()), ('pca', PCA(n_components=0.99, svd_solver='full')), ('cf', LinearSVC())])

    params = {
        'cf__C': [1.0],
        'cf__loss': ['squared_hinge'],
        'cf__dual': [False],
    }

    print('Grid: ', params)

    print('Scorer: ', scorer)

    cf = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring=scorer, refit=scorer[0])

    start = time.time()
    cf.fit(X_train, y_train)
    end = time.time()
    print('SVC cross-val time elapsed: ', end - start)

    print('Best params: ', cf.best_params_)

    print('PCA number of components', cf.best_estimator_.named_steps['pca'].n_components_)

    balanced_acc_score = cf.best_score_ * 100

    acc_score = cf.cv_results_['mean_test_accuracy'][cf.best_index_] * 100

    print("Best cross-val balanced accuracy score: " + str(round(balanced_acc_score, 2)) + '%')
    print("Best cross-val accuracy score: " + str(round(acc_score, 2)) + '%')

    cv_results = pd.DataFrame(cf.cv_results_)
    # display(cv_results)

    print('\n')
    return cf.best_estimator_
