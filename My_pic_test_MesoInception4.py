# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from classifiers import MesoInception4

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0].name}")
else:
    print("No GPU available, using CPU.")

# Parameters
IMG_WIDTH = 256
MODEL_WEIGHTS_PATH = 'weights/MesoInception_DF.h5'  # Relative path to weights
IMAGE_DIR_REAL = 'images_real'  # Relative path to image directory
IMAGE_DIR_FAKE = 'images_fake'  # Relative path to image directory

# Initialize model
model = MesoInception4(learning_rate=0.001)
model.load(MODEL_WEIGHTS_PATH)

# Load and preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_WIDTH, IMG_WIDTH))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# Predict and print results for each image
def judge_image(image_path):
    print(f"\nAnalyzing images in {image_path}:")
    print("-" * 60)
    
    for image_name in os.listdir(image_path):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join(image_path, image_name)
            img_data = preprocess_image(file_path)
            prediction = model.predict(img_data)[0][0]
            label = 'Real' if prediction >= 0.5 else 'Fake'
            print(f'Image: {image_name:30} | Prediction: {prediction:.4f} | Label: {label}')
    
    print("-" * 60)

print("\nTesting MesoInception4 model on images...")
judge_image(IMAGE_DIR_REAL)
judge_image(IMAGE_DIR_FAKE)
print("\nAnalysis complete!")