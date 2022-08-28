import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/regression_models/decision_trees/Position_Salaries.csv")
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=0)
dt_model.fit(X,y)

print(dt_model.predict([[6.5]]))

# # Plot the decision tree regression result
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color='red')
# plt.plot(X_grid, dt_model.predict(X_grid))
# plt.show()
