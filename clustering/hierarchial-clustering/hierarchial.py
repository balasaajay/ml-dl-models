import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/clustering/k-means/Mall_Customers.csv")
X = data.iloc[:,[3,4]].values

# # Using dendrogramsto to find optimal clusters
# import scipy.cluster.hierarchy as sch
# # methos of minimum variants - clusters inside which variance is low (ward)
# dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
# plt.title('Dendrogram')
# plt.xlabel('customers')
# plt.ylabel('Eucledian distance')
# plt.show()

# Based on the Dendrogram - 5 clusters are optimum
# train the model
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred = model.fit_predict(X)
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
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()
