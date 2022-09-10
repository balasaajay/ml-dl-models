# Recommender system
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("dimensionality_reduction/principal_component_analysis/Wine.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test) 

# Implement PCA:
from sklearn.decomposition import PCA
pca = PCA(n_components=2) # Number of output features-start with 2 and increment 1 at a time for best results
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Log reg
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print(np.concatenate(y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1), 1))

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))