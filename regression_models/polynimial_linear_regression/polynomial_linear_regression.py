import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/regression_models/polynimial_linear_regression/Position_Salaries.csv")
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
# No need of first column as it is redundant with second
# No need to split of data as we have less data

from sklearn.linear_model import LinearRegression

# Linear regression
model_lin = LinearRegression()
model_lin.fit(X, y)

# Polynomial regression implemnetation
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
# print(X_poly)
model_poly = LinearRegression()
model_poly.fit(X_poly,y)

# # Plot the model
# plt.scatter(X, y)
# plt.plot(X, model_lin.predict(X), color = 'red')
# plt.plot(X, model_poly.predict(poly_reg.fit_transform(X)), color = 'blue')
# plt.show()

# Predict:
test_data = [[6.5]]
test_poly_data = poly_reg.fit_transform(test_data)
print(model_lin.predict(test_data))
print(model_poly.predict(test_poly_data))
