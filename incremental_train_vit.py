# -*- coding: utf-8 -*-
# filepath: /mnt/d/python_code/MesoNet/incremental_train_vit.py

import os
import torch
import numpy as np
from PIL import Image
import PIL
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from classifiers import ViTClassifier

# Parameters
IMG_WIDTH = 256
BATCH_SIZE = 16  # Smaller batch size suitable for incremental training
EPOCHS = 5       # Fewer epochs needed for incremental training
LEARNING_RATE = 1e-5  # Lower learning rate to prevent overfitting
WEIGHTS_PATH = 'weights/vit_finetuned'  # Path to existing weights
TRAIN_FOLDER = 'train'  # Original training folder with real/fake subdirectories

# Set device to use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create image transformations
transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Properly handle image loading errors
def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = PIL.Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        # Return a blank image as fallback
        return PIL.Image.new('RGB', (IMG_WIDTH, IMG_WIDTH))

# Load dataset and create data loaders
def create_dataloader(folder, img_width, batch_size):
    # Check if folder exists and contains the expected subfolders
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Training folder '{folder}' not found")
    
    # Check real and fake directories
    real_dir = os.path.join(folder, 'real')
    fake_dir = os.path.join(folder, 'fake')
    
    if not os.path.exists(real_dir) or not os.listdir(real_dir):
        raise FileNotFoundError(f"No images found in '{real_dir}'. Please add real images to this directory.")
    
    if not os.path.exists(fake_dir) or not os.listdir(fake_dir):
        raise FileNotFoundError(f"No images found in '{fake_dir}'. Please add fake images to this directory.")
    
    print(f"Found {len(os.listdir(real_dir))} real images and {len(os.listdir(fake_dir))} fake images")
    
    # Create dataset from folder
    dataset = ImageFolder(folder, transform=transform, loader=pil_loader)
    
    # Create 80/20 train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Created training data loader with {train_size} images")
    print(f"Created validation data loader with {val_size} images")
    
    return train_loader, val_loader

# Perform incremental training
def incremental_train():
    print("Starting incremental training for ViT model...")
    
    # Check if weights path exists
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Warning: Weights path {WEIGHTS_PATH} does not exist. Will use pre-trained model for incremental training.")
    
    # Initialize ViT model
    model = ViTClassifier(model_name='google/vit-base-patch16-224', num_labels=2)
    
    # Ensure model is on GPU if available
    model.device = device
    model.model = model.model.to(device)
    print(f"Model moved to device: {model.device}")
    
    # Try to load existing weights
    if os.path.exists(WEIGHTS_PATH):
        try:
            # Load model from path with the appropriate device
            model.model = model.model.from_pretrained(WEIGHTS_PATH)
            model.model = model.model.to(device)
            print(f"Successfully loaded model weights from {WEIGHTS_PATH}!")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Continuing with base pre-trained model...")
    
    # Create data loaders from the train folder
    train_loader, val_loader = create_dataloader(TRAIN_FOLDER, IMG_WIDTH, BATCH_SIZE)
    
    # Train with early stopping
    # Lower threshold and higher patience for incremental training
    accuracy_threshold = 0.98
    patience = 3
    
    # Start training
    print(f"Starting incremental training with learning rate {LEARNING_RATE}...")
    trained_model = train_vit_with_early_stopping(
        model, 
        train_loader, 
        val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE,
        accuracy_threshold=accuracy_threshold,
        patience=patience
    )
    
    # Save model, overwriting original weights file
    print(f"Saving incrementally trained model to {WEIGHTS_PATH}...")
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    trained_model.model.save_pretrained(WEIGHTS_PATH)
    print("Model saved successfully!")
    
    return trained_model

# Modified training function (adapted from train_xception_efficientnet.py)
def train_vit_with_early_stopping(vit_model, train_loader, val_loader, epochs=5, lr=1e-5, 
                                  accuracy_threshold=0.98, patience=3):
    """
    Train ViT model with early stopping mechanism for incremental learning
    
    Parameters:
        vit_model: ViT model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of training epochs
        lr: Learning rate
        accuracy_threshold: Early stopping accuracy threshold
        patience: Number of epochs to wait before early stopping
    
    Returns:
        best_model: The best trained model
    """
    
    optimizer = torch.optim.AdamW(vit_model.model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    best_accuracy = 0.0
    no_improve_count = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        vit_model.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            images, labels = batch
            
            # Convert to format needed by feature extractor
            pil_images = []
            for img in images:
                # Denormalize: img * std + mean
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                # Convert to PIL format
                img = img.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).clip(0, 255).astype('uint8')
                pil_images.append(img)
            
            # Process features
            inputs = vit_model.feature_extractor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(vit_model.device) for k, v in inputs.items()}
            
            # Convert labels
            labels = labels.to(vit_model.device)
            
            # Forward pass
            outputs = vit_model.model(**inputs, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_count % 5 == 0:  # Report progress more frequently
                print(f"Incremental Training: Epoch {epoch+1}/{epochs}, Batch {batch_count}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")
        
        # Validation phase
        if val_loader:
            current_accuracy = vit_model.evaluate(val_loader)
            
            # Check if accuracy threshold is reached
            if current_accuracy >= accuracy_threshold:
                print(f"Target accuracy {accuracy_threshold:.4f} reached, stopping training early")
                best_model_state = vit_model.model.state_dict().copy()
                break
            
            # Check if accuracy has improved
            if current_accuracy > best_accuracy:
                print(f"Validation accuracy improved from {best_accuracy:.4f} to {current_accuracy:.4f}")
                best_accuracy = current_accuracy
                best_model_state = vit_model.model.state_dict().copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"Validation accuracy did not improve, {no_improve_count}/{patience} epochs without improvement")
                
                # Check if patience limit is reached
                if no_improve_count >= patience:
                    print(f"No improvement for {patience} consecutive epochs, stopping training early")
                    break
    
    # If best model was found, restore that state
    if best_model_state is not None:
        print(f"Restoring best model state, accuracy: {best_accuracy:.4f}")
        vit_model.model.load_state_dict(best_model_state)
    
    return vit_model

# Main execution
if __name__ == "__main__":
    # Perform incremental training
    incremental_train()
    print("Incremental training completed!")