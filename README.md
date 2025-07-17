# GAN Image Generation: Pistachio Dataset

A personal deep learning project exploring the architecture and training dynamics of Generative Adversarial Networks (GANs) through the generation of synthetic pistachio images.

## Project Overview

This repository contains a comprehensive implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) designed to generate realistic pistachio images. The project was undertaken as an independent study to deepen understanding of GAN architectures, adversarial training dynamics, and the challenges associated with training generative models.

### Objectives

The primary objectives of this project include:
- Understanding the theoretical foundations of GANs and their practical implementation
- Exploring the training dynamics between generator and discriminator networks
- Investigating the impact of hyperparameter choices on training stability and output quality
- Gaining hands-on experience with deep learning frameworks and GPU-accelerated training

## Dataset

The project utilises the **Pistachio Image Dataset** sourced from Kaggle, specifically focusing on the Kirmizi Pistachio variety (red pistachios). This dataset was selected for its moderate complexity, providing a suitable balance between challenging feature learning and computational tractability.

### Dataset Details
- **Source**: [Kaggle Pistachio Image Dataset](https://www.kaggle.com/datasets/muratkoklu/pistachio-image-dataset)
- **Variety**: Kirmizi Pistachio (red pistachio variety)
- **Format**: JPG images with standardised naming convention
- **Preprocessing**: Images resized to 128×128 pixels and normalised to [-1, 1] range

## Technical Approach

### Architecture

The implementation follows the Deep Convolutional GAN (DCGAN) architecture with the following key components:

**Generator Network**:
- Input: 75-dimensional latent noise vector
- Architecture: Series of transposed convolutions with batch normalisation
- Activation: LeakyReLU for hidden layers, Tanh for output layer
- Output: 128×128 RGB images

**Discriminator Network**:
- Input: 128×128 RGB images (real or generated)
- Architecture: Series of convolutional layers with batch normalisation
- Activation: LeakyReLU for hidden layers, Sigmoid for output layer
- Output: Single probability score (real vs fake classification)

### Training Process

The training employs a systematic approach to hyperparameter exploration:

1. **Adversarial Training**: Alternating optimisation of discriminator and generator networks
2. **Hyperparameter Sweep**: Systematic exploration of learning rate combinations
3. **Loss Function**: Binary cross-entropy loss for both networks
4. **Optimisation**: Adam optimiser
5. **Monitoring**: Real-time loss tracking and periodic image generation

## Repository Structure

```
GAN_project/
├── README.md                           # Project documentation
├── GAN_dataset_new copy.ipynb          # Main training notebook
├── environment.yml                     # Conda environment specification
├── .gitignore                         # Git ignore patterns
└── dataset/                           # Dataset directory (structure tracked)
    ├── DATASET_INFO.md                # Dataset setup instructions
    ├── Pistachio_Image_Dataset/       # Dataset directory
    │   ├── README.md                  # Dataset setup instructions
    │   ├── Kirmizi_Pistachio/         # Red pistachio images (*.jpg excluded from Git)
    │   └── Pistachio_Image_Dataset_Request.txt # Dataset citation info
    └── CW_GAN/                        # Generated images and training outputs
        ├── README.md                  # Output directory information
        ├── real_samples.png           # Sample real images (excluded from Git)
        ├── fake_samples_*.png         # Generated images by epoch (excluded from Git)
        ├── GAN_G_model.pth            # Saved generator model (excluded from Git)
        ├── GAN_D_model.pth            # Saved discriminator model (excluded from Git)
        └── loss_data.pkl              # Training loss history (excluded from Git)
```

## Usage Instructions

### Environment Setup

**Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate fyp-dev
   ```

### Running the Code

1. **Navigate to the project directory**:
   ```bash
   cd GAN_project
   ```

2. **Open the main notebook**:
   ```bash
   jupyter notebook "GAN_dataset_new copy.ipynb"
   ```

3. **Execute the cells sequentially**:
   - Import libraries and set hyperparameters
   - Load and preprocess the dataset
   - Define the GAN architecture
   - Run the training loop with hyperparameter sweep

### Key Files

- **`GAN_dataset_new copy.ipynb`**: Main training notebook with comprehensive documentation
- **`environment.yml`**: Complete dependency specification for reproducibility
- **`dataset/DATASET_INFO.md`**: Dataset setup instructions and information

## Training Notes and Limitations

### Training Environment

The model training was initiated using GPU infrastructure at École Polytechnique, providing access to CUDA-enabled hardware for accelerated training. The implementation is designed to automatically detect and utilise available GPU resources while maintaining CPU compatibility.

However, due to academic timeline constraints and the conclusion of access to École Polytechnique's supercomputing facilities, comprehensive hyperparameter optimization could not be completed. This limitation represents a significant constraint on the project's ability to achieve optimal model performance and convergence.

### Known Limitations

Due to time constraints and infrastructure access limitations at École Polytechnique, the following constraints apply:

1. **Incomplete Hyperparameter Optimization**: Optimal hyperparameter combinations have not been identified due to project timeline constraints and limited access to supercomputing resources before departure from École Polytechnique
2. **Incomplete Training**: Not all hyperparameter combinations were fully explored to convergence
3. **Limited Epochs**: Some configurations may benefit from extended training beyond 100 epochs
4. **Infrastructure Dependency**: Optimal training requires access to high-performance computing resources for systematic hyperparameter exploration
5. **Memory Constraints**: Batch size selection depends on available GPU memory

### Training Outputs

The training process generates:
- **Generated Images**: Timestamped samples showing generator progress
- **Model Checkpoints**: Saved Generator and Discriminator state dictionaries
- **Loss History**: Comprehensive tracking of training dynamics
- **Hyperparameter Logs**: Systematic record of all tested configurations

## Results and Observations

The project successfully demonstrates:
- **Functional GAN Implementation**: Both Generator and Discriminator networks train successfully
- **Training Framework**: Comprehensive hyperparameter sweep methodology implemented
- **Progressive Improvement**: Generated images show increasing quality over training epochs
- **Training Dynamics**: Proper adversarial balance between Generator and Discriminator losses