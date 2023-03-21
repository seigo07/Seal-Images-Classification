import pandas as pd
import seaborn as sns
import time

from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


# Loading data
def load_data():
    print('Loading data...')

    df = {
        'X_train': pd.read_csv("binary/X_train.csv", header=None),
        'y_train': pd.read_csv("binary/Y_train.csv", header=None),
        'X_test': pd.read_csv("binary/X_test.csv", header=None)
    }
    df['X_train'].info()
    print('Unique values', df['y_train'].iloc[:, 0].unique())
    df['X_test'].info()

    return df


# Cleaning the data by removing the features taken from a random normal distribution
def clean_data(data_dict):
    print('Cleaning data...')
    df = {
        'X_train': data_dict['X_train'].drop(data_dict['X_train'].columns[900:916], axis=1),
        'X_test': data_dict['X_test'].drop(data_dict['X_test'].columns[900:916], axis=1),
        'y_train': data_dict['y_train']
    }
    print('X train after cleaning:')
    df['X_train'].info()
    print('X_test after cleaning')
    df['X_test'].info()
    return df


# Visualising the target frequencies to check the imbalance of data.
def plot_target_frequency(df):
    df['y_train'].columns = ['label']
    print(df['y_train'].columns)
    y_train_size = float(len(df['y_train']))
    print('y_train_size: ', y_train_size)
    plot = sns.countplot(x='label', data=df['y_train'])
    for patch in plot.patches:
        plot.text(patch.get_x() + patch.get_width() / 2.,
                  patch.get_height() + 3,
                  '{:1.3f}'.format(patch.get_height() / y_train_size),
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
# The estimator is a pipeline in which the features are scaled.
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


# Defining the estimator as well as the grid search. The estimator is a pipeline in which the features are scaled.
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
