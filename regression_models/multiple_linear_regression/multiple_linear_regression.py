import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/regression_models/multiple_linear_regression/50_Startups.csv")
X = data.iloc[1:, :-1].values
y = data.iloc[1:, -1].values
# print(X)

# Encoding categorical values
# in X using oneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)


# Split data to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
# print(X_train, X_test, y_train, y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train[:,3:] = std_scaler.fit_transform(X_train[:,3:]) # first 3 columns are already in -3,3 range

# Use the same scaler to transform test set data
X_test[:,3:] = std_scaler.transform(X_test[:,3:]) 
# print(X_train, X_test, y_train, y_test)


# Train
# this LinearRegression class has support for Dummy variable trap and also picks best features for statistical significance (backward elimination)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

np.set_printoptions(precision=2)
# axis=1 - hostizontal concatenation, axis=0 - vertical concat
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)),axis=1))
# print(model.coef_)
# print(model.intercept_)

# RootMSE and meanAbsolute error are popular
mse = mean_squared_error(y_test,  y_pred)
rmse = mean_squared_error(y_test,  y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
print(mse, rmse, mae)