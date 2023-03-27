import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, \
    classification_report, f1_score


RANDOM_STATE = 42
N_SPLITS = 5

# Loading data
def load_data(classfication_type):
    print('Loading data...')
    df = {
        'X_train': pd.read_csv(classfication_type+"/X_train.csv", header=None),
        'y_train': pd.read_csv(classfication_type+"/Y_train.csv", header=None),
        'X_test': pd.read_csv(classfication_type+"/X_test.csv", header=None)
    }
    df['X_train'].info()
    print('Unique values', df['y_train'].iloc[:, 0].unique())
    df['X_test'].info()

    return df


# Cleaning the data by removing the features taken from a random normal distribution
def clean_data(df):
    print('Cleaning data...')
    df = {
        'X_train': df['X_train'].drop(df['X_train'].columns[900:916], axis=1),
        'X_test': df['X_test'].drop(df['X_test'].columns[900:916], axis=1),
        'y_train': df['y_train']
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


# cross_validation function for KNeighborsClassifier.
def kn_cross_validation(X_train, y_train, scorer):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline(
        [('sc', StandardScaler()), ('pca', PCA(n_components=0.99, svd_solver='full')), ('cf', KNeighborsClassifier())])

    params = {
        'cf__n_neighbors': [3],
        'cf__leaf_size': [30],
        'cf__p': [2]
    }

    print('Params for GridSearchCV: ', params)
    print('Scorer: ', scorer)

    gs = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring=scorer, refit=scorer[0])
    start = time.time()
    gs.fit(X_train, y_train)
    end = time.time()

    print('The time of cross-validation for KNeighborsClassifier: ', end - start)
    print('Best parameters for GridSearchCV: ', gs.best_params_)
    print('The number of components of PCA', gs.best_estimator_.named_steps['pca'].n_components_)

    balanced_acc_score = gs.best_score_ * 100
    acc_score = gs.cv_results_['mean_test_accuracy'][gs.best_index_] * 100

    print("Best cross-validation balanced accuracy score: " + str(round(balanced_acc_score, 2)) + '%')
    print("Best cross-validation accuracy score: " + str(round(acc_score, 2)) + '%')

    # cv_results = pd.DataFrame(gs.cv_results_)
    # display(cv_results)

    return gs.best_estimator_


# cross_validation function for RandomForestClassifier.
def rf_cross_validation(X_train, y_train, scorer):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline(
        [('sc', StandardScaler()), ('pca', PCA(n_components=0.99, svd_solver='full')),
         ('cf', RandomForestClassifier())])

    params = {
        'cf__n_estimators': [400],
        'cf__max_depth': [None],
        'cf__min_samples_split': [2],
    }

    print('Params for GridSearchCV: ', params)
    print('Scorer: ', scorer)

    gs = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring=scorer, refit=scorer[0])

    start = time.time()
    gs.fit(X_train, y_train)
    end = time.time()

    print('The time of cross-validation for Random Forest: ', end - start)
    print('Best parameters for GridSearchCV: ', gs.best_params_)
    print('The number of components of PCA', gs.best_estimator_.named_steps['pca'].n_components_)

    balanced_acc_score = gs.best_score_ * 100
    acc_score = gs.cv_results_['mean_test_accuracy'][gs.best_index_] * 100

    print("Best cross-validation balanced accuracy score: " + str(round(balanced_acc_score, 2)) + '%')
    print("Best cross-validation accuracy score: " + str(round(acc_score, 2)) + '%')

    # cv_results = pd.DataFrame(cf.cv_results_)
    # display(cv_results)

    return gs.best_estimator_


# cross_validation function for LinearSVC.
def svc_cross_validation(X_train, y_train, scorer):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline(
        [('sc', StandardScaler()), ('pca', PCA(n_components=0.99, svd_solver='full')), ('cf', LinearSVC())])

    params = {
        'cf__C': [1.0],
        'cf__loss': ['squared_hinge'],
        'cf__dual': [False],
    }

    print('Params for GridSearchCV: ', params)
    print('Scorer: ', scorer)

    gs = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring=scorer, refit=scorer[0])

    start = time.time()
    gs.fit(X_train, y_train)
    end = time.time()

    print('The time of cross-validation for SVC: ', end - start)
    print('Best parameters for GridSearchCV: ', gs.best_params_)
    print('The number of components of PCA', gs.best_estimator_.named_steps['pca'].n_components_)

    balanced_acc_score = gs.best_score_ * 100
    acc_score = gs.cv_results_['mean_test_accuracy'][gs.best_index_] * 100

    print("Best cross-validation balanced accuracy score: " + str(round(balanced_acc_score, 2)) + '%')
    print("Best cross-validation accuracy score: " + str(round(acc_score, 2)) + '%')

    # cv_results = pd.DataFrame(cf.cv_results_)
    # display(cv_results)

    return gs.best_estimator_


def printResult(y, pred):
    print('classification accuracy = ', accuracy_score(y, pred))
    print('balanced accuracy = ', balanced_accuracy_score(y, pred))
    print('confusion matrix = \n', confusion_matrix(y, pred))
    print('precision micro = ', precision_score(y, pred, average='micro'))
    print('precision macro = ', precision_score(y, pred, average='macro'))
    print('recall micro = ', recall_score(y, pred, average='micro'))
    print('recall macro = ', recall_score(y, pred, average='macro'))
    print('f1_score macro = ', f1_score(y, pred, average='micro'))
    print('f1_score macro = ', f1_score(y, pred, average='macro'))
    print('classification report: \n', classification_report(y, pred))


def outputCSV(y_test, classfication_type, classfier):
    np.savetxt("output/"+classfication_type+"/"+classfier+"/Y_test.csv", y_test, fmt="%s")
