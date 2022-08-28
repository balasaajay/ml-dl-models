import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/regression_models/decision_trees/Position_Salaries.csv")
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor
# n_estimators - number of trees to build
# randome_state - for repeatability
rf_model = RandomForestRegressor(n_estimators=30, random_state=0)
rf_model.fit(X,y)

print(rf_model.predict([[6.5]]))

# # Plot the decision tree regression result
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color='red')
# plt.plot(X_grid, rf_model.predict(X_grid))
# plt.show()
