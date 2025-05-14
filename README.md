# Deepfake-Facial-Image-Detection

An experimental study on detecting deepfake facial images using deep learning models: MesoInception4 model, followed by finetuned Xception, EfficientNetB0, and Vision Transformer (ViT) models, trained on a Kaggle deepfake dataset. Results demonstrate that fine-tuned models significantly outperform others.

## Version 1.1 Updates

- **GPU Acceleration**: All models now support GPU/CUDA for both training and inference processes
- **Model Training Scripts**: Training processes have been separated into individual scripts for easier maintenance and usage
- **Enhanced Inference**: Improved inference scripts with better error handling and performance reporting
- **Code Refactoring**: Restructured codebase for better maintainability and readability
- **Large Model Support**: Weight files larger than 100MB are now stored in GitHub Releases
- **Incremental Training**: Added support for continuing training from existing checkpoints

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

### Incremental Training

The `incremental_train_vit.py` script allows you to continue training from a previously saved model checkpoint. This is useful when you:
- Have new training data to incorporate
- Want to fine-tune the model further
- Need to resume interrupted training sessions

Usage example:
```bash
python incremental_train_vit.py
```

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

## Model Weights

Due to GitHub's file size limitations, model weight files larger than 100MB are stored in the GitHub Releases section. You can download them from:
- [GitHub Releases](https://github.com/DaweiYuNYIT/Deepfake-Facial-Image-Detection/releases)

The following weight files are available in the releases:
- Vision Transformer (ViT) model weights (> 100MB)
- Other large model checkpoints

After downloading, place the weight files in the `weights/` directory:
- For ViT weights: `weights/vit_finetuned/`
- For other models: directly in `weights/`

## Performance

Fine-tuned models (particularly Xception) have demonstrated superior performance in detecting modern deepfakes, including those from advanced AI systems like Gemini and Grok.
