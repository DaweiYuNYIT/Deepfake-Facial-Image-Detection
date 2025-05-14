# -*- coding: utf-8 -*-
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from classifiers import ViTClassifier
from PIL import Image
import torch

# Parameters
IMG_WIDTH = 256
XCEPTION_WEIGHTS_PATH = 'weights/xception_finetuned.weights.h5'  # Path to Xception weights
EFFICIENTNET_WEIGHTS_PATH = 'weights/efficientnet_finetuned.weights.h5'  # Path to EfficientNet weights
VIT_WEIGHTS_PATH = 'weights/vit_finetuned'  # Path to ViT weights
IMAGE_DIR_REAL = 'images_real'  # Directory with real images
IMAGE_DIR_FAKE = 'images_fake'  # Directory with fake images

# Build Xception model
def build_xception_model():
    base_model = Xception(weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_WIDTH, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=y)
    return model

# Build EfficientNetB0 model
def build_efficientnet_model():
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_WIDTH, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=y)
    return model

# Build ViT model
def build_vit_model():
    return ViTClassifier()

# Load and preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_WIDTH, IMG_WIDTH))
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Predict and print results for each image using a given model
def judge_image(model, model_name, image_dir, true_label):
    print(f"\nTesting {model_name} on {image_dir} (True label: {true_label})")
    correct = 0
    total = 0
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.jpg', '.png')):
            file_path = os.path.join(image_dir, image_name)
            img_data = preprocess_image(file_path)
            prediction = model.predict(img_data, verbose=0)[0][0]
            predicted_label = 'Real' if prediction >= 0.5 else 'Fake'
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct += 1
            total += 1
            print(f'Path: {file_path}, Prediction: {prediction:.4f}, Label: {predicted_label}')
    accuracy = correct / total if total > 0 else 0
    print(f'{model_name} Accuracy on {image_dir}: {accuracy:.4f} ({correct}/{total})')

# Predict and print results for each image using ViT model
def judge_image_vit(model, model_name, image_dir, true_label):
    print(f"\nTesting {model_name} on {image_dir} (True label: {true_label})")
    correct = 0
    total = 0
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.jpg', '.png')):
            file_path = os.path.join(image_dir, image_name)
            img = Image.open(file_path).convert('RGB')
            prediction = model.predict(np.array(img))
            predicted_label = 'Real' if prediction >= 0.5 else 'Fake'
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct += 1
            total += 1
            print(f'Path: {file_path}, Prediction: {prediction:.4f}, Label: {predicted_label}')
    accuracy = correct / total if total > 0 else 0
    print(f'{model_name} Accuracy on {image_dir}: {accuracy:.4f} ({correct}/{total})')

# Initialize and load models
xception_model = build_xception_model()
xception_model.load_weights(XCEPTION_WEIGHTS_PATH) 
efficientnet_model = build_efficientnet_model()
efficientnet_model.load_weights(EFFICIENTNET_WEIGHTS_PATH)
vit_model = build_vit_model()
try:
    vit_model.model.load_state_dict(torch.load(VIT_WEIGHTS_PATH + '/pytorch_model.bin'))
except Exception as e:
    print('ViT weights not loaded:', e)

# Test models on real and fake images
judge_image(xception_model, 'Xception', IMAGE_DIR_REAL, 'Real')
judge_image(xception_model, 'Xception', IMAGE_DIR_FAKE, 'Fake')
judge_image(efficientnet_model, 'EfficientNetB0', IMAGE_DIR_REAL, 'Real')
judge_image(efficientnet_model, 'EfficientNetB0', IMAGE_DIR_FAKE, 'Fake')
judge_image_vit(vit_model, 'ViT', IMAGE_DIR_REAL, 'Real')
judge_image_vit(vit_model, 'ViT', IMAGE_DIR_FAKE, 'Fake')