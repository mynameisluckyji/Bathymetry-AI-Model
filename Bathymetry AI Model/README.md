# DLProject 
# PINN-SSL-MMF: Deep Learning for Bathymetry Estimation

This repository provides three advanced deep learning pipelines for bathymetry estimation using remote sensing imagery. The models leverage multimodal data, self-supervised learning, and physics-inspired neural networks to enable robust depth prediction from aerial, satellite, and multispectral inputs.
## 📚 Table of Contents

[Project Structure](https://github.com/Rish-1903/DLProject/tree/main?tab=readme-ov-file#-project-structure)

[Model Descriptions](https://github.com/Rish-1903/DLProject/blob/main/README.md#-model-descriptions)

[Data Preparation](https://github.com/Rish-1903/DLProject?tab=readme-ov-file#-data-preparation)

[Usage](https://github.com/Rish-1903/DLProject?tab=readme-ov-file#-usage)

[Evaluation & Visualization](https://github.com/Rish-1903/DLProject?tab=readme-ov-file#-usage)

[Requirements](https://github.com/Rish-1903/DLProject?tab=readme-ov-file#-requirements)

[Citation](https://github.com/Rish-1903/DLProject?tab=readme-ov-file#-citation)

## 📁 Project Structure
```bash
.
├── multimodal_transformer.py      # Multimodal Transformer for bathymetry
├── selfsupervised_patchnet.py     # Self-Supervised PatchNet for feature learning
├── physics_inspired_nn.py         # Physics-Inspired Neural Network (PINN)
├── requirements.txt
├── README.md
└── (data folders and output directories)

```
 


## 🧠 Model Descriptions

### 1. Multimodal Transformer for Bathymetry


Fuses aerial, Sentinel-2, and SPOT6 imagery for robust depth prediction.

Architecture: Shared CNN encoder → Transformer blocks → CNN decoder.

Supports gradient accumulation, mixed-precision training, and model checkpointing.

Outputs predicted depth maps at 256×256 resolution.

### 2. Self-Supervised

Learns image representations by reconstructing random patches from SPOT6 images.

Architecture: Convolutional autoencoder.

Ideal for pretraining or extracting features when depth labels are unavailable.

Includes patch extraction, data augmentation, and visualization utilities.

### 3. Physics-Informed Neural Network (PINN)

Predicts bathymetry from SPOT6 images using physics-consistent loss functions.

Architecture: Residual CNN blocks with channel/spatial attention and skip connections.

Loss components: Data-driven, smoothness, and depth-consistency terms.

Supports sliding window inference for large scenes and advanced augmentations.

## 📦 Data Preparation

Organize your data as follows:
```bash
MagicBathyNet/agia_napa/
├── img/
│   ├── aerial/
│   ├── s2/
│   └── spot6/
├── depth/
│   ├── aerial/
│   ├── s2/
│   └── spot6/
├── norm_param_aerial.npy
├── norm_param_s2_an.npy
└── norm_param_spot6_an.npy
```
Ensure that the normalization parameter .npy files are present for each image modality.

## 🚀 Usage
### 1. Train Multimodal Transformer

    python multimodalfusion.py

Trains the transformer-based model using all modalities.

Automatically saves the best model as best_model.pth.

Evaluates on the test set after training and prints performance metrics.

### 2. Train Self-Supervised PatchNet

    python selfsupervised.py

Trains the PatchNet autoencoder using SPOT6 patches.

Saves reconstruction outputs and visualization images for test examples.

### 3. Train Physics-Inspired Neural Network (PINN)

    python physics_inspired_nn.py

Trains the PINN model with physics-based loss terms.

Performs early stopping, and saves model checkpoints.

Evaluates on the test set and saves predictions and comparison plots.

## 📊 Evaluation & Visualization

All scripts report standard evaluation metrics: MAE, MSE, RMSE, SSIM, etc.

Built-in visualization tools display:

    Input images

    Ground truth depths

    Predicted depth maps

Sample outputs:

    reconstruction_*.tif

    prediction_*.tif

    comparison_*.png

## 📦 Requirements

Refer to requirements.txt for dependencies:

    torch
    torchvision
    numpy
    matplotlib
    scikit-image
    tqdm
    rasterio
    einops
    gdal


## 📌 Citation
[MagicBathyNet](https://github.com/pagraf/MagicBathyNet)

