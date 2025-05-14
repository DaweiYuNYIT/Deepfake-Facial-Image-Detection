# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using TensorFlow GPU: {physical_devices[0].name}")
else:
    print("No TensorFlow GPU available, using CPU for training.")

# Parameters
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 8
WEIGHTS_PATH = 'weights/xception_finetuned.weights.h5'

def create_data_generators():
    # Create data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        'train',  # Directory with 'real' and 'fake' subfolders
        target_size=(IMG_WIDTH, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'  # Binary classification (real vs fake)
    )

    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        'validate',  # Directory with 'real' and 'fake' subfolders
        target_size=(IMG_WIDTH, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    return train_generator, validation_generator

# Build and fine-tune Xception
def build_xception_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_WIDTH, 3))
    base_model.trainable = False  # Freeze the base model layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=y)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_xception():
    print("\nStarting Xception model training...")
    
    # Create data generators
    train_generator, validation_generator = create_data_generators()
    print(f"Found {train_generator.samples} training images in {len(train_generator.class_indices)} classes")
    print(f"Found {validation_generator.samples} validation images in {len(validation_generator.class_indices)} classes")
    
    # Build model
    xception_model = build_xception_model()
    print("Xception model built successfully")
    
    # Train model
    print(f"\nTraining Xception model for {EPOCHS} epochs...")
    history = xception_model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Save model weights
    os.makedirs('weights', exist_ok=True)
    xception_model.save_weights(WEIGHTS_PATH)
    print(f"Xception model weights saved to {WEIGHTS_PATH}")
    
    # Print final metrics
    final_loss, final_accuracy = xception_model.evaluate(validation_generator)
    print(f"Final validation accuracy: {final_accuracy:.4f}")
    print(f"Final validation loss: {final_loss:.4f}")
    
    return xception_model, history

# Execute training when run directly
if __name__ == "__main__":
    train_xception()