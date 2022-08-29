from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/classification_models/Social_Network_Ads.csv")
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

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=40,criterion='entropy',random_state = 0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print(np.concatenate(y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1), 1))

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# # Visualising the Training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = std_scaler.inverse_transform(X_train), y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
#                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# plt.contourf(X1, X2, clf.predict(std_scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()