import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/clustering/k-means/Mall_Customers.csv")
X = data.iloc[:, [3,4]].values

# Use elbow method to identify optimal number of clusters:
from sklearn.cluster import KMeans
# wcss = list()
# for i in range(1, 11):
#   kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
#   kmeans.fit(X)
#   wcss.append(kmeans.inertia_)

# plt.plot(range(1, 11), wcss, color='red')
# plt.title('Elbow method')
# plt.xlabel('num of clusters')
# plt.ylabel('wcss score')
# plt.show()

# based on the chart - 5 is the best number

# Train the model with 5 clusters
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = kmeans.fit_predict(X)   # Create a dependent variable to split the data - fit_predict witll return the clusters
# print(y_pred)

# Visualizing the clusters
plt.scatter(X[ y_pred == 0 , 0], \
  X[ y_pred == 0 , 1], s=100, \
  c = 'cyan', label = 'Cluster 1') # Select the rows that belong to cluster 0
plt.scatter(X[ y_pred == 1 , 0], \
  X[ y_pred == 1, 1], s=100, \
  c = 'blue', label = 'Cluster 2') # Select the rows that belong to cluster 0
plt.scatter(X[ y_pred == 2 , 0], \
  X[ y_pred == 2 , 1], s=100, \
  c = 'green', label = 'Cluster 3') # Select the rows that belong to cluster 0
plt.scatter(X[ y_pred == 3 , 0], \
  X[ y_pred == 3, 1], s=100, \
  c = 'orange', label = 'Cluster 4') # Select the rows that belong to cluster 0
plt.scatter(X[ y_pred == 4 , 0], \
  X[ y_pred == 4, 1], s=100, \
  c = 'purple', label = 'Cluster 5') # Select the rows that belong to cluster 0
# Plot the centroid from y_pred
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], \
  s=300, c='magenta', label='centroids')
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()