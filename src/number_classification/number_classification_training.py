import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Load images and corresponding annotations
def load_data(dataset_path):
    images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            images.append(image)
            label = int(filename.split('_')[0])  # Extract label from filename
            labels.append(label)
    return np.array(images), np.array(labels)

# Prepare dataset
dataset_path = 'enhanced_dataset'
images, labels = load_data(dataset_path)

# Split dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape images to match input shape expected by CNN
input_shape = (train_images.shape[1], train_images.shape[2], 1)  # Assuming grayscale images
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Assuming 10 classes (numbers 0-9)
])

# Trained model location
model_dir = '../models/cnn_classification'

# Save the trained model
model.save(os.path.join(model_dir, 'my_model'))

print("Model saved successfully.")

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}')
