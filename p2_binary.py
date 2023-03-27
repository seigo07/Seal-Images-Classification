from p2_functions import *
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# 1. Loading the data

df = load_data("binary")

# 2. Cleaning the data and creating new input features from the given dataset

df = clean_data(df)

# 3. Analysing and visualising the data

plot_target_frequency(df)

# 4. Preparing the inputs and choosing a suitable subset of features

# Splitting the training dataset to get a mock test dataset for the X_test.csv

X_train, X_test, y_train, y_test = train_test_split(df['X_train'], df['y_train'],
                                                    random_state=RANDOM_STATE, shuffle=True, stratify=df['y_train'])

# Under-sampling to deal with imbalanced data.

X_train_under, y_train_under = RandomUnderSampler(random_state=0).fit_resample(X_train, y_train)
df_under = {
    'y_train': y_train_under
}
plot_target_frequency(df_under)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()
X_train_under = X_train_under.to_numpy()
y_train_under = y_train_under.to_numpy().ravel()

# 5. Selecting and training a classification model,

# Using cross-validation and predicting the performance of KNeighborsClassifier.

scorer = ['balanced_accuracy', 'accuracy']
kn = kn_cross_validation(X_train, y_train, scorer)
y_pred = kn.predict(X_test)

# Evaluating the mis-classification of each class the confusion matrix and the classification report.

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
printResult(y_test, y_pred)

# Training and evaluating KNN with under-sampled data.

kn_under = kn_cross_validation(X_train_under, y_train_under, scorer=scorer)
y_pred_under = kn_under.predict(X_test)
cf_under = confusion_matrix(y_test, y_pred_under)
sns.heatmap(cf_under / np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')
printResult(y_test, y_pred_under)

# 6. Selecting and training another classification model

# Using cross-validation and predicting the performance of Random Forest.

rf = rf_cross_validation(X_train, y_train, scorer=scorer)
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')

# Evaluating the mis-classification of each class the confusion matrix and the classification report.

printResult(y_test, y_pred)

# Training and evaluating RF with under-sampled data.

rf_under = rf_cross_validation(X_train_under, y_train_under, scorer=scorer)
y_pred_under = rf_under.predict(X_test)
cf_under = confusion_matrix(y_test, y_pred_under)
sns.heatmap(cf_under / np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')
printResult(y_test, y_pred_under)

# Using cross-validation and predicting the performance of LinearSVC.

svc = svc_cross_validation(X_train, y_train, scorer=scorer)
y_pred = svc.predict(X_test)

# Evaluating the mis-classification of each class the confusion matrix and the classification report.

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
printResult(y_test, y_pred)

# Training and evaluating SVC with under-sampled data.

svc_under = svc_cross_validation(X_train_under, y_train_under, scorer=scorer)
y_pred_under = svc_under.predict(X_test)
cf_under = confusion_matrix(y_test, y_pred_under)
sns.heatmap(cf_under / np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')
printResult(y_test, y_pred_under)

# Producing the Y_test.csv file which is used to evaluate the final performance of the classifier (SVM classifier).

outputCSV(kn.predict(df['X_test']), "binary", "kn")
outputCSV(rf.predict(df['X_test']), "binary", "rf")
outputCSV(svc.predict(df['X_test']), "binary", "svc")
