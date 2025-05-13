# -*- coding: utf-8 -*-
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from classifiers import Meso4  # meso_models.py in the same directory

# Parameters
IMG_WIDTH = 256
MODEL_WEIGHTS_PATH = 'weights/Meso4_DF.h5'  # Relative path to weights
IMAGE_DIR_REAL = 'images_real'  # Relative path to image directory
IMAGE_DIR_FAKE = 'images_fake'  # Relative path to image directory

# Initialize model
model = Meso4(learning_rate=0.001)
model.load(MODEL_WEIGHTS_PATH)

# Load and preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_WIDTH, IMG_WIDTH))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# Predict and print results for each image
def judge_image(image_path):
    for image_name in os.listdir(image_path):
        if image_name.endswith(('.jpg', '.png')):
            file_path = os.path.join(image_path, image_name)
            img_data = preprocess_image(file_path)
            prediction = model.predict(img_data)[0][0]
            label = 'Real' if prediction >= 0.5 else 'Fake'
            print(f'Path:{file_path},Prediction: {prediction:.4f}, Label: {label}')

judge_image(IMAGE_DIR_REAL)
judge_image(IMAGE_DIR_FAKE)