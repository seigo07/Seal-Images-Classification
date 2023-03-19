
from p2_multi_functions import *
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score


# 1. Loading the data

data_dict = load_data()

# 2. Cleaning the data and creating new input features from the given dataset

data_dict = clean_data(data_dict)

# 3. Analysing and visualising the data

plot_target_frequency(data_dict)

# 4. Preparing the inputs and choosing a suitable subset of features

# Splitting the training dataset in order to obtain a mock test set,
# due to no target labels for the X_test.csv file.

X_train, X_val, y_train, y_val = train_test_split(data_dict['X_train'], data_dict['y_train'], test_size=0.2, random_state=23, shuffle=True, stratify=data_dict['y_train'])

# Getting the undersampled data for training without undersampling the mock testing set
# in order to keep the distribution of the classes close to the distribution of the original dataset.

rus = RandomUnderSampler(random_state=0)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
data_dict_under = {
    'y_train': y_train_under
}
plot_target_frequency(data_dict_under)

# Converting the dataframes into numpy ndarrays.

X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
y_train = y_train.to_numpy().ravel()
y_val = y_val.to_numpy().ravel()
X_train_under = X_train_under.to_numpy()
y_train_under = y_train_under.to_numpy().ravel()

# Defining the scoring metrics for cross-validation.

scorer = ['balanced_accuracy', 'accuracy']

knn_cf = kn_cross_validate_pca(X_train, y_train, scorer)

# Making predictions on the validation set and evaluate the classifier's performance.

y_pred = knn_cf.predict(X_val)

# The confusion matrix for KNN which will tell us how each class was misclassified.

cf = confusion_matrix(y_val, y_pred)
sns.heatmap(cf/np.sum(cf), annot=True, fmt='.2%', cmap='Blues')

# The classification report for KNN which will give us detailed evaluation metrics.

cr = classification_report(y_val, y_pred)
print(cr)

print('Balanced accuracy: ', balanced_accuracy_score(y_val, y_pred))

# Training KNN with undersampled dataset.

knn_cf_under = kn_cross_validate_pca(X_train_under, y_train_under, scorer)

# Making predictions on evaluation set with KNN trained on the undersampled dataset
# by Using the KNN classifier trained using the undersampled dataset.

y_pred_under = knn_cf_under.predict(X_val)

# The confusion matrix which will tell us how each class was misclassified.

cf_under = confusion_matrix(y_val, y_pred_under)
sns.heatmap(cf_under/np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')

# The classification report for KNN which will give us detailed evaluation metrics.

cr_under = classification_report(y_val, y_pred_under)
print(cr_under)

# 5. Selecting and training a classification model,

rf_cf = rf_cross_validate_pca(X_train, y_train, scorer)

# Making predictions on the validation set with RF by using the RF classifier
# to make predictions on the validation set to allow us to evaluate its performance.

y_pred = rf_cf.predict(X_val)

# The confusion matrix for RF which will tell us how each class was misclassified.

cf = confusion_matrix(y_val, y_pred)
sns.heatmap(cf/np.sum(cf), annot=True, fmt='.2%', cmap='Blues')

# The classification report for RF which will give us detailed evaluation metrics.

cr = classification_report(y_val, y_pred)
print(cr)

print('Balanced accuracy: ', balanced_accuracy_score(y_val, y_pred))

# Training RF with undersampled dataset using the undersampled dataset.

rf_cf_under = rf_cross_validate_pca(X_train_under, y_train_under, scorer)

# Making prediction on evaluation set with RF using the RF classifier trained using the undersampled dataset
# to make predictions on the validation set to allow us to evaluate its performance.

y_pred_under = rf_cf_under.predict(X_val)

# The confusion matrix for RF which will tell us how each class was misclassified.

cf_under = confusion_matrix(y_val, y_pred_under)
sns.heatmap(cf_under/np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')

# The classification report for RF which will give us detailed evaluation metrics.

cr_under = classification_report(y_val, y_pred_under)
print(cr_under)

# 6. Selecting and training another classification model

svc_cf = svc_cross_validate_pca(X_train, y_train, scorer)

# Making predictions on the validation set with SVC using the SVC classifier
# to make predictions on the validation set to allow  us to evaluate its performance.

y_pred = svc_cf.predict(X_val)

# The confusion matrix for SVC which will tell us how each class was misclassified.

cf = confusion_matrix(y_val, y_pred)
sns.heatmap(cf/np.sum(cf), annot=True, fmt='.2%', cmap='Blues')
cr = classification_report(y_val, y_pred)
print(cr)
print('Balanced accuracy: ', balanced_accuracy_score(y_val, y_pred))

# Training SVC with undersampled dataset

svc_cf_under = svc_cross_validate_pca(X_train_under, y_train_under, scorer)

# Making prediction on evaluation set with SVC under using the SVC classifier trained using the undersampled dataset
# to make predictions on the validation set to allow  us to evaluate its performance.

y_pred_under = svc_cf_under.predict(X_val)

# The confusion matrix for SVC which will tell us how each class was misclassified.

cf_under = confusion_matrix(y_val, y_pred_under)
sns.heatmap(cf_under/np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')

# The classification report for SVC which will give us detailed evaluation metrics.

cr_under = classification_report(y_val, y_pred_under)
print(cr_under)

# Producing the Y_test.csv file which is the file that will be used to evaluate the final performance of the classifier.
# Decided that will be used for the SVM classifier that was trained on the original dataset after analysing the results.

X_test = data_dict['X_test']
y_test = svc_cf.predict(X_test)
print(len(y_test))
# np.savetxt("output/Y_test.csv", y_test, fmt="%s")



















