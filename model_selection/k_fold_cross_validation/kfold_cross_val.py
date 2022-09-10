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

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=clf, X=X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()* 100))
print("Std dev: {:.2f} %".format(accuracies.std()* 100))