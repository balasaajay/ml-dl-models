import numpy as np
import pandas as pd
import tensorflow as tf

# print(tf.__version__) 
# Read data from the dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/Deep Learning/artificial_neural_network/Churn_Modelling.csv")
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values

# Encoding categorical values in X
# Gender column - Label encoder
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
X[:, 2] = label_enc.fit_transform(X[:, 2])

# Country column - One hot encoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X.info)
# Split data to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
  
# Feature Scaling - absolutely needed for feature scaling
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train) # first 3 columns are already in -3,3 range
X_test = std_scaler.transform(X_test) 

# Construct Neural network
ann_model = tf.keras.models.Sequential()
# Add input layer, hidden layers
ann_model.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann_model.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Add output layer
# sigmoid output activation gives probablities of the output being 1
ann_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Train the model
# 1. Compile model - set optimizer and loss functions
# adam optimizer is the best one that performs stochastic gradient descent
# loss function - since this is a binary classification, use binary_crossentropy
ann_model.compile(optimizer='adam', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# 2. Train using batch values
ann_model.fit(X_train, y_train, batch_size=32, epochs = 100)

# Predict for the data:
print(ann_model.predict(std_scaler.transform([[1, 0, 0, 600 , 1, 40, 3, 60000, 2, 1, 1, 50000]])))
# output = [[0.05439119]]
print(ann_model.predict(std_scaler.transform([[1, 0, 0, 600 , 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5 )

# Collect model metrics
y_pred = ann_model.predict(X_test)
y_pred = (y_pred > 0.5)
loss, accuracy = ann_model.evaluate(X_test, y_test)
print(loss, accuracy)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
