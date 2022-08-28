import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/regression_models/support_vector_regression_svr/Position_Salaries.csv")
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
y = y.reshape(len(y), 1)  # convert list to np 2D array

# Feature scaling is needed for SVR
# 
from sklearn.preprocessing import StandardScaler
std_scaler_X = StandardScaler()
X = std_scaler_X.fit_transform(X) # first 3 columns are already in -3,3 range

# We cannot use same scalar for X and y
std_scaler_y = StandardScaler()
y = np.squeeze(std_scaler_y.fit_transform(y))

# Train the model
from sklearn.svm import SVR
model = SVR(kernel="rbf") # Using gaussian RBF
model.fit(X, y)

test = np.array([[6.5]])
print(std_scaler_y.inverse_transform(([model.predict(std_scaler_X.transform(test))])))

# # Plot the model
# plt.scatter(std_scaler_X.inverse_transform(X), \
#   std_scaler_y.inverse_transform(y.reshape(len(y), 1)))
# plt.plot(std_scaler_X.inverse_transform(X), std_scaler_y.inverse_transform(model.predict(X)), color = 'red')
# plt.show()
