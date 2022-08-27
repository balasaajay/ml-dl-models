import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("Data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Impute missing data with Mean
from sklearn.impute import SimpleImputer
simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 1:3] = simple_imputer.fit_transform(X[:, 1:3])

# Encoding categorical values
# in X using oneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# For Y (labels)
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)
print(y)

# Split data to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
# print(X_train, X_test, y_train, y_test)

# Feature Scaling
# This step is always done after train/test split because
# 1. test set is brand new set to evaluate model
# 2. Feature scaling may get mean/std deviation of all the test set values causes INFORMATION LEAKAGE
# Two techniqes:
# 1. Standardization - outputs values in range (-3, 3) x = (x - mean(x))/std_dev(x) - works all the time
# 2. Normalisation - outputs vals in (0,1) x = (x-min(x))/(max(x)-min(x)) - Recommended when features are in normal(gaussian) distribution
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train[:,3:] = std_scaler.fit_transform(X_train[:,3:]) # first 3 columns are already in -3,3 range

# Use the same scaler to transform test set data
X_test[:,3:] = std_scaler.transform(X_test[:,3:]) 
print(X_train, X_test, y_train, y_test)
