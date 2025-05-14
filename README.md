# Deepfake-Facial-Image-Detection

An experimental study on detecting deepfake facial images using deep learning models: MesoInception4 model, followed by finetuned Xception, EfficientNetB0, and Vision Transformer (ViT) models, trained on a Kaggle deepfake dataset. Results demonstrate that fine-tuned models significantly outperform others.

## Version 1.1 Updates

- **GPU Acceleration**: All models now support GPU/CUDA for both training and inference processes
- **Model Training Scripts**: Training processes have been separated into individual scripts for easier maintenance and usage
- **Enhanced Inference**: Improved inference scripts with better error handling and performance reporting
- **Code Refactoring**: Restructured codebase for better maintainability and readability

## Model Architecture

The project employs multiple deep learning architectures to detect deepfake images:

1. **MesoNet Architectures**:
   - Meso4: A CNN-based model designed specifically for deepfake detection
   - MesoInception4: An inception-based variant with improved performance

2. **Transfer Learning Models**:
   - Xception: A CNN architecture with depthwise separable convolutions
   - EfficientNetB0: A compact and efficient CNN architecture
   - Vision Transformer (ViT): A transformer-based model for image classification

## Training Scripts

Each model can now be trained independently using the following scripts:

- `train_xception.py`: Trains the Xception model with ImageNet pre-trained weights
- `train_efficientnet.py`: Trains the EfficientNetB0 model with ImageNet pre-trained weights
- `train_vit.py`: Trains the Vision Transformer model with early stopping capability
- `incremental_train_vit.py`: Continues training an existing ViT model with new data

All training scripts utilize GPU acceleration (if available) and include progress reporting and weight saving functionality.

## Inference Scripts

- `My_pic_test_Meso4.py`: Tests images using the Meso4 model
- `My_pic_test_MesoInception4.py`: Tests images using the MesoInception4 model
- `My_pic_test_vit.py`: Tests images using the Vision Transformer model
- `My_pic_test_xception_efficientnet.py`: Tests images using both Xception and EfficientNetB0 models

## Setup and Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Training a model (e.g., Xception):
   ```bash
   python train_xception.py
   ```

3. Testing images:
   ```bash
   python My_pic_test_xception_efficientnet.py
   ```

Each training and inference script is configured to use GPU/CUDA acceleration when available, with automatic fallback to CPU processing.

## Dataset Structure

The dataset should be organized as follows:
- `train/real/`: Real images for training
- `train/fake/`: Deepfake images for training
- `validate/real/`: Real images for validation
- `validate/fake/`: Deepfake images for validation

## Performance

Fine-tuned models (particularly Xception) have demonstrated superior performance in detecting modern deepfakes, including those from advanced AI systems like Gemini and Grok.
