# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from PIL import Image
from classifiers import ViTClassifier

# Parameters
IMG_WIDTH = 256
MODEL_WEIGHTS_PATH = 'weights/vit_finetuned'  # Path to saved ViT model weights
IMAGE_DIR_REAL = 'images_real'  # Relative path to real image directory
IMAGE_DIR_FAKE = 'images_fake'  # Relative path to fake image directory

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
print("Loading ViT model...")
model = ViTClassifier(model_name='google/vit-base-patch16-224', num_labels=2)

# Ensure model is on the CUDA device if available
model.device = device
model.model = model.model.to(device)

# Load weights if they exist
if os.path.exists(MODEL_WEIGHTS_PATH):
    try:
        model.model = model.model.from_pretrained(MODEL_WEIGHTS_PATH)
        model.model = model.model.to(device)  # Ensure weights are on the CUDA device
        print(f"Model loaded successfully from {MODEL_WEIGHTS_PATH}!")
    except Exception as e:
        print(f"Error loading weights from {MODEL_WEIGHTS_PATH}: {e}")
        print("Continuing with the base pre-trained model...")
else:
    print(f"Warning: Weight path {MODEL_WEIGHTS_PATH} not found. Using base pre-trained model.")

print(f"Model is on device: {model.device}")

# Load and preprocess images
def preprocess_image(image_path):
    # Open and resize image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_WIDTH))
    return img

# Predict and print results for each image
def judge_image(image_path):
    print(f"\nAnalyzing images in {image_path}:")
    print("-" * 60)
    
    for image_name in os.listdir(image_path):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join(image_path, image_name)
            try:
                img = preprocess_image(file_path)
                
                # Get prediction
                with torch.no_grad():  # Disable gradient calculation for inference
                    prediction = model.predict(img)
                
                # For binary classification
                label = 'Real' if prediction > 0.5 else 'Fake'
                
                print(f'Image: {image_name:30} | Prediction: {prediction:.4f} | Label: {label}')
            except Exception as e:
                print(f'Error processing {file_path}: {str(e)}')
                import traceback
                traceback.print_exc()  # Print full traceback for debugging
    
    print("-" * 60)

# Test on real and fake images
print("\nTesting ViT model on images...")
judge_image(IMAGE_DIR_REAL)
judge_image(IMAGE_DIR_FAKE)
print("\nAnalysis complete!")