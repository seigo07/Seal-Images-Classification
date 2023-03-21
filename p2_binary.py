from p2_binary_functions import *
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

# 1. Loading the data

df = load_data()

# 2. Cleaning the data and creating new input features from the given dataset

df = clean_data(df)

# 3. Analysing and visualising the data

plot_target_frequency(df)

# 4. Preparing the inputs and choosing a suitable subset of features

# Splitting the training dataset to get a mock test dataset for the X_test.csv

X_train, X_test, y_train, y_test = train_test_split(df['X_train'], df['y_train'], test_size=0.2,
                                                    random_state=23, shuffle=True, stratify=df['y_train'])

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

# Defining the scoring metrics for cross-validation.

scores = ['balanced_accuracy', 'accuracy']

knn_cf = kn_cross_validate_pca(X_train, y_train, scores)

# Making predictions on the validation set and evaluate the classifier's performance.

y_pred = knn_cf.predict(X_test)

# The confusion matrix for KNN which will tell us how each class was misclassified.

cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf / np.sum(cf), annot=True, fmt='.2%', cmap='Blues')

# The classification report for KNN which will give us detailed evaluation metrics.

cr = classification_report(y_test, y_pred)
print(cr)

print('Balanced accuracy: ', balanced_accuracy_score(y_test, y_pred))

# Training KNN with undersampled dataset.

knn_cf_under = kn_cross_validate_pca(X_train_under, y_train_under, scorer=scores)

# Making predictions on evaluation set with KNN trained on the undersampled dataset
# by Using the KNN classifier trained using the undersampled dataset.

y_pred_under = knn_cf_under.predict(X_test)

# The confusion matrix which will tell us how each class was misclassified.

cf_under = confusion_matrix(y_test, y_pred_under)
sns.heatmap(cf_under / np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')

# The classification report for KNN which will give us detailed evaluation metrics.

cr_under = classification_report(y_test, y_pred_under)
print(cr_under)

# 5. Selecting and training a classification model,

rf_cf = rf_cross_validate_pca(X_train, y_train, scorer=scores)

# Making predictions on the validation set with RF by using the RF classifier
# to make predictions on the validation set to allow us to evaluate its performance.

y_pred = rf_cf.predict(X_test)

# The confusion matrix for RF which will tell us how each class was misclassified.

cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf / np.sum(cf), annot=True, fmt='.2%', cmap='Blues')

# The classification report for RF which will give us detailed evaluation metrics.

cr = classification_report(y_test, y_pred)
print(cr)

print('Balanced accuracy: ', balanced_accuracy_score(y_test, y_pred))

# Training RF with undersampled dataset using the undersampled dataset.

rf_cf_under = rf_cross_validate_pca(X_train_under, y_train_under, scorer=scores)

# Making prediction on evaluation set with RF using the RF classifier trained using the undersampled dataset
# to make predictions on the validation set to allow us to evaluate its performance.

y_pred_under = rf_cf_under.predict(X_test)

# The confusion matrix for RF which will tell us how each class was misclassified.

cf_under = confusion_matrix(y_test, y_pred_under)
sns.heatmap(cf_under / np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')

# The classification report for RF which will give us detailed evaluation metrics.

cr_under = classification_report(y_test, y_pred_under)
print(cr_under)

# 6. Selecting and training another classification model

svc_cf = svc_cross_validate_pca(X_train, y_train, scorer=scores)

# Making predictions on the validation set with SVC using the SVC classifier
# to make predictions on the validation set to allow  us to evaluate its performance.

y_pred = svc_cf.predict(X_test)

# The confusion matrix for SVC which will tell us how each class was misclassified.

cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf / np.sum(cf), annot=True, fmt='.2%', cmap='Blues')
cr = classification_report(y_test, y_pred)
print(cr)
print('Balanced accuracy: ', balanced_accuracy_score(y_test, y_pred))

# Training SVC with undersampled dataset

svc_cf_under = svc_cross_validate_pca(X_train_under, y_train_under, scorer=scores)

# Making prediction on evaluation set with SVC under using the SVC classifier trained using the undersampled dataset
# to make predictions on the validation set to allow  us to evaluate its performance.

y_pred_under = svc_cf_under.predict(X_test)

# The confusion matrix for SVC which will tell us how each class was misclassified.

cf_under = confusion_matrix(y_test, y_pred_under)
sns.heatmap(cf_under / np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')

# The classification report for SVC which will give us detailed evaluation metrics.

cr_under = classification_report(y_test, y_pred_under)
print(cr_under)

# Producing the Y_test.csv file which is the file that will be used to evaluate the final performance of the classifier.
# Decided that will be used for the SVM classifier that was trained on the original dataset after analysing the results.

# X_test = df['X_test']
# y_test = svc_cf.predict(X_test)
# np.savetxt("output/Y_test.csv", y_test, fmt="%s")
