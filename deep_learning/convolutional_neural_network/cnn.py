import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# print(tf.__version__) 
# Read data from the dataset

# Preprocess trainset
# refer to https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255, # Feature scaling
        shear_range=0.2, # Shear angle in counter-clockwise direction in degrees
        zoom_range=0.2, # Range for random zoom
        horizontal_flip=True # Randomly flip inputs horizontally
        )
train_set = train_datagen.flow_from_directory(
        '/Users/abalasa/github/balasaajay/ml-model-templates/deep_learning/convolutional_neural_network/dataset/training_set',
        target_size=(64, 64), # Final size of image fed to CNN
        batch_size=32, # How many images in each batch
        class_mode='binary' # Binary outcome - cats/dogs?
        )

# Preprocess test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        '/Users/abalasa/github/balasaajay/ml-model-templates/deep_learning/convolutional_neural_network/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Build the CNN model

# Initialize the model
cnn_model = tf.keras.models.Sequential()
# Add convolutional layer
# Filters - How many feature detectors?
# Kernel size - 3X3
cnn_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))
# Add pooling layer
# pool_size - How many pixels to consider pooling
# strides - How many pixels to skip for next pooling operation
cnn_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Add second convolutional layer and pool layer
cnn_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) # Removed input_shape as it changes
cnn_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Flattening - convert to 1D vector
cnn_model.add(tf.keras.layers.Flatten())
# Full connectected layers:
# units: Number of neurons
cnn_model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# sigmoid output activation gives probablities of the output being 1
cnn_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Train the CNN
# Compiling the model
cnn_model.compile(optimizer='adam', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
cnn_model.fit(
        x = train_set,
        epochs=25,
        validation_data=test_set)

# Predict on predict images
import numpy as np
test_image = tf.keras.preprocessing.image.load_img(
  '/Users/abalasa/github/balasaajay/ml-model-templates/deep_learning/convolutional_neural_network/dataset/single_prediction/cat_or_dog_1.jpg',
  target_size=(64, 64)
  ) # returns PIL image format
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0) # Convert image to batches
predictions = cnn_model.predict(test_image)
print(predictions)

print(train_set.class_indices)
if predictions[0][0] == 1:
  pred = 'Dog'
else:
  pred = 'Cat'
print(pred)
