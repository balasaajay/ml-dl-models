import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("Salary_Data.csv")
X = data.iloc[1:, :-1].values
y = data.iloc[1:, -1].values

# Impute missing data with Mean
from sklearn.impute import SimpleImputer
simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, :] = simple_imputer.fit_transform(X[:, :])

# Split data to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
# print(X_train, X_test, y_train, y_test)

# Train the Linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
# print(y_pred, y_test)

# Plot for training data and regressor
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience - Training set")
plt.xlabel("Experience in yrs")
plt.ylabel("Salary")
plt.show()

# Plot for testing data and regressor
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience - Testing set")
plt.xlabel("Experience in yrs")
plt.ylabel("Salary")
plt.show()


# test prediction
print(regressor.predict([[12]]))

# Print coefficients
# Salary=coef×YearsExperience+intercept
# Salary=9345.94×YearsExperience+26816.19
print(regressor.coef_)
print(regressor.intercept_)
