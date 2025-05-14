# -*- coding: utf-8 -*-
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
BATCH_SIZE = 32
EPOCHS = 8
WEIGHTS_PATH = 'weights/vit_finetuned'

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

def create_dataloader(folder, img_width, batch_size):
    """Create PyTorch data loaders for ViT training."""
    # Check if folder exists and contains the expected subfolders
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Training folder '{folder}' not found")
    
    # Create dataset from folder
    dataset = ImageFolder(folder, transform=transform, loader=pil_loader)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Created dataloader for {folder} with {len(dataset)} images")
    
    return dataloader

def build_vit_model():
    """Initialize a ViT model for binary classification."""
    # Initialize ViT classifier configured for binary classification
    model = ViTClassifier(model_name='google/vit-base-patch16-224', num_labels=2)
    
    # Ensure model is on the correct device
    model.device = device
    model.model = model.model.to(device)
    
    return model

def train_vit_with_early_stopping(vit_model, train_loader, val_loader, epochs=8, lr=2e-5, 
                                  accuracy_threshold=0.95, patience=3):
    """
    Train ViT model with early stopping mechanism
    
    Parameters:
        vit_model: ViT model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of training epochs
        lr: Learning rate
        accuracy_threshold: Early stopping accuracy threshold, training stops when validation accuracy reaches this value
        patience: Tolerance count, training stops if validation accuracy doesn't improve for this many consecutive epochs
    
    Returns:
        best_model: The best trained model
    """
    optimizer = torch.optim.AdamW(vit_model.model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    best_accuracy = 0.0
    no_improve_count = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        vit_model.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            images, labels = batch
            # Convert normalized tensor images to PIL images for feature extractor
            pil_images = []
            for img in images:
                # Denormalize: img * std + mean
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                # Convert to PIL
                img = img.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).clip(0, 255).astype('uint8')
                pil_images.append(img)
            
            # Process with feature extractor
            inputs = vit_model.feature_extractor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(vit_model.device) for k, v in inputs.items()}
            
            # Convert labels
            labels = labels.to(vit_model.device)
            
            # Forward pass
            outputs = vit_model.model(**inputs, labels=labels)
            loss = outputs.loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_count}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")
        
        # Validation phase, calculate accuracy
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
                
                # Check if patience is reached
                if no_improve_count >= patience:
                    print(f"No improvement for {patience} consecutive epochs, stopping training early")
                    break
    
    # If best model was found, restore that state
    if best_model_state is not None:
        print(f"Restoring best model state, accuracy: {best_accuracy:.4f}")
        vit_model.model.load_state_dict(best_model_state)
    
    return vit_model

def train_vit():
    """Main function to train the ViT model."""
    print("\nStarting ViT model training...")
    
    # Create data loaders
    print("Creating data loaders for ViT training...")
    train_loader = create_dataloader('train', IMG_WIDTH, BATCH_SIZE)
    val_loader = create_dataloader('validate', IMG_WIDTH, BATCH_SIZE)
    
    # Build ViT model
    print("Building ViT model...")
    vit_model = build_vit_model()
    print(f"ViT model built and moved to device: {vit_model.device}")
    
    # Early stopping parameters
    accuracy_threshold = 0.97  # Stop training at 97% accuracy
    patience = 2  # Stop after 2 epochs without improvement
    
    # Train with early stopping
    print(f"\nTraining ViT model for up to {EPOCHS} epochs with early stopping...")
    trained_vit_model = train_vit_with_early_stopping(
        vit_model, 
        train_loader, 
        val_loader, 
        epochs=EPOCHS, 
        lr=2e-5,
        accuracy_threshold=accuracy_threshold,
        patience=patience
    )
    
    # Save model
    print(f"Saving ViT model to {WEIGHTS_PATH}...")
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    trained_vit_model.model.save_pretrained(WEIGHTS_PATH)
    print("ViT model saved successfully.")
    
    return trained_vit_model

# Execute training when run directly
if __name__ == "__main__":
    train_vit()