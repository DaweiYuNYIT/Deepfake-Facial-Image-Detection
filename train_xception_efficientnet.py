import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Parameters
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 8

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255
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

# Build and fine-tune Xception
def build_xception_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_WIDTH, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=y)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# xception_model = build_xception_model()
# xception_model.fit(
#     train_generator,
#     validation_data=validation_generator,
#     epochs=EPOCHS
# )
# xception_model.save_weights('weights/xception_finetuned.weights.h5')

# Build and fine-tune EfficientNet
def build_efficientnet_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_WIDTH, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=y)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

efficientnet_model = build_efficientnet_model()
efficientnet_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)
efficientnet_model.save_weights('weights/efficientnet_finetuned.weights.h5')