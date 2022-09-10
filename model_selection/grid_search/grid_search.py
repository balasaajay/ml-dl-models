import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/model_selection/Social_Network_Ads.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)
# print(X_train, X_test, y_train, y_test)

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)
# print(X_test)

from sklearn.svm import SVC
clf = SVC(kernel='rbf', random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Kfold
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=clf, X=X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()* 100))
print("Std dev: {:.2f} %".format(accuracies.std()* 100))

# Grid search
from sklearn.model_selection import GridSearchCV
parameters = [{
  'C': [0.25, 0.5,0.75, 1],  # List of different values to try
  'kernel': ['linear']
}, {
  'C': [0.25, 0.5,0.75, 1],  # List of different values to try
  'kernel': ['rbf', 'poly'],
  'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}]
grid_search = GridSearchCV(estimator=clf, param_grid=parameters,scoring='accuracy',
              cv=10, n_jobs=-1)

grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
print("Best Params: {}".format(best_parameters))
# Best Accuracy: 90.67 %
# Best Params: {'C': 0.5, 'gamma': 0.6, 'kernel': 'rbf'}
