import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os
from tqdm import tqdm
from PIL import Image

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load dataset
data_dir = 'C://Users//sukar//Desktop//Plant Watering and Monitoring System//plant-pathology-2020-fgvc7//images'  # Update with the path to your images directory
metadata = pd.read_csv('C://Users//sukar//Desktop//Plant Watering and Monitoring System//plant-pathology-2020-fgvc7//train.csv')  # Update with the path to your train.csv

# Data preprocessing
image_size = (128, 128)

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize(image_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image

# Load images and labels
images = []
labels = []

for idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
    img_path = os.path.join(data_dir, row['image_id'] + '.jpg')
    images.append(load_and_preprocess_image(img_path))
    labels.append(row['label'])

images = np.array(images)
labels = np.array(labels)

# Encode labels
label_encoder = LabelBinarizer()
labels = label_encoder.fit_transform(labels)

# Split data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator()

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
batch_size = 32
epochs = 50

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=val_datagen.flow(X_val, y_val),
    epochs=epochs,
    callbacks=[early_stopping]
)

# Save the model
model.save('plant_disease_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
