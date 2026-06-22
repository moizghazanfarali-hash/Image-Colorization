# Image Colorization Model

Black-and-white image colorization using a GAN-based deep learning model (PyTorch).

## Overview
- **Generator**: ECCV16-based colorization network (`color_ecv`)
- **Discriminator**: PatchGAN discriminator
- **Feature Extractor**: Pretrained VGG19 (for perceptual/content loss)
- **Color Space**: LAB (L channel = input grayscale, AB channels = predicted color)
- **Dataset**: [Image Colorization Dataset](https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset) (via `kagglehub`)

## Requirements
```
torch
torchvision
numpy
pillow
matplotlib
scikit-image
kagglehub
gradio
```

## Workflow
1. Download dataset using `kagglehub`
2. Preprocess images (RGB → LAB, resize to 400x400)
3. Build `ImageDataset` (loads black/orig/color image triplets)
4. Define Generator, Discriminator, and VGG19 Feature Extractor
5. Train using GAN loss (MSE) + content loss (L1) via Adam optimizer
6. Save model checkpoints in `colorit_gan/saved_models/`
7. Run inference through a **Gradio web app** for interactive colorization

## Usage
Run all notebook cells in order. After training, the Gradio app launches automatically, letting you upload a grayscale image and get a colorized output.

## Notes
- Set `n_epochs` and `batch_size` as per your GPU/resources.
- Checkpoints can be resumed by setting `start_epoch` > 0.
