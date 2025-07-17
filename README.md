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

### Citation
The dataset is based on the following research:
- OZKAN IA., KOKLU M. and SARACOGLU R. (2021). Classification of Pistachio Species Using Improved K-NN Classifier. Progress in Nutrition, Vol. 23, N. 2.
- SINGH D, TASPINAR YS, KURSUN R, CINAR I, KOKLU M, OZKAN IA, LEE H-N., (2022). Classification and Analysis of Pistachio Species with Pre-Trained Deep Learning Models, Electronics, 11 (7), 981.

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
4. **Optimisation**: Adam optimiser with β₁ = 0.5, β₂ = 0.999
5. **Monitoring**: Real-time loss tracking and periodic image generation

### Key Hyperparameters

- **Learning Rates**: Generator (0.00001 to 0.01), Discriminator (0.000001 to 0.001)
- **Batch Size**: 256 samples per batch
- **Epochs**: 100 training epochs per configuration
- **Latent Dimension**: 75-dimensional noise vector
- **Architecture Scale**: 20 base channels with progressive scaling

## Tools and Technologies

The project leverages the following technologies:

- **PyTorch**: Deep learning framework for model implementation and training
- **torchvision**: Image processing and transformation utilities
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Visualisation of training progress and generated samples
- **PIL (Pillow)**: Image loading and basic processing
- **tqdm**: Progress bar visualisation for training loops
- **Git**: Version control with automated commit integration

## Repository Structure

```
GAN_project/
├── README.md                           # Project documentation
├── environment.yml                     # Conda environment specification
├── .gitignore                         # Git ignore patterns
└── src/
    └── dataset/
        ├── GAN_dataset.py             # Core GAN implementation (classes and functions)
        ├── GAN_dataset_new copy.ipynb # Main training notebook
        ├── Pistachio_Image_Dataset/   # Dataset directory
        │   ├── Kirmizi_Pistachio/     # Red pistachio images
        │   ├── Siirt_Pistachio/       # Alternative variety (unused)
        │   └── Pistachio_Image_Dataset_Request.txt # Dataset citation info
        └── CW_GAN/                    # Generated images and training outputs
            ├── real_samples.png       # Sample real images
            ├── fake_samples_*.png     # Generated images by epoch
            ├── GAN_G_model.pth        # Saved generator model
            ├── GAN_D_model.pth        # Saved discriminator model
            └── loss_data.pkl          # Training loss history
```

## Usage Instructions

### Environment Setup

1. **Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate fyp-dev
   ```

2. **Verify PyTorch installation**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True if GPU is available
   ```

### Running the Code

1. **Navigate to the project directory**:
   ```bash
   cd GAN_project
   ```

2. **Open the main notebook**:
   ```bash
   jupyter notebook src/dataset/GAN_dataset_new\ copy.ipynb
   ```

3. **Execute the cells sequentially**:
   - Import libraries and set hyperparameters
   - Load and preprocess the dataset
   - Define the GAN architecture
   - Run the training loop with hyperparameter sweep

### Key Files

- **`GAN_dataset_new copy.ipynb`**: Main training notebook with comprehensive documentation
- **`GAN_dataset.py`**: Standalone implementation of GAN classes and utilities
- **`environment.yml`**: Complete dependency specification for reproducibility

## Training Notes and Limitations

### Training Environment

The model training was initiated using GPU infrastructure at École Polytechnique, providing access to CUDA-enabled hardware for accelerated training. The implementation is designed to automatically detect and utilise available GPU resources while maintaining CPU compatibility.

### Known Limitations

Due to access limitations with the GPU infrastructure during the project completion phase, the following constraints apply:

1. **Incomplete Training**: Not all hyperparameter combinations were fully explored to convergence
2. **Limited Epochs**: Some configurations may benefit from extended training beyond 100 epochs
3. **Hardware Dependency**: Optimal training requires GPU acceleration for reasonable training times
4. **Memory Constraints**: Batch size selection depends on available GPU memory

### Training Outputs

The training process generates:
- **Generated Images**: Timestamped samples showing generator progress
- **Model Checkpoints**: Saved Generator and Discriminator state dictionaries
- **Loss History**: Comprehensive tracking of training dynamics
- **Hyperparameter Logs**: Systematic record of all tested configurations

## Results and Observations

The project successfully demonstrates:
- **Functional GAN Implementation**: Both Generator and Discriminator networks train successfully
- **Hyperparameter Sensitivity**: Clear impact of learning rate choices on training stability
- **Progressive Improvement**: Generated images show increasing quality over training epochs
- **Training Dynamics**: Proper adversarial balance between Generator and Discriminator losses

## Future Enhancements

Potential improvements for future iterations include:
- **Extended Training**: Longer training with more epochs for optimal convergence
- **Architecture Refinements**: Exploration of more sophisticated GAN architectures (e.g., StyleGAN, Progressive GAN)
- **Evaluation Metrics**: Implementation of quantitative evaluation measures (FID, IS scores)
- **Data Augmentation**: Enhanced preprocessing for improved dataset diversity
- **Hyperparameter Optimisation**: Automated hyperparameter search using modern optimisation techniques

## Acknowledgements

This project was developed as an independent study to explore generative adversarial networks. Special acknowledgement to École Polytechnique for providing the GPU infrastructure that enabled the initial training experiments.

The dataset creators (Ozkan, Koklu, Saracoglu, et al.) deserve recognition for making the Pistachio Image Dataset publicly available, enabling research and educational applications in computer vision and machine learning.

---

*This project represents a comprehensive exploration of GAN architectures and training dynamics, providing valuable insights into the challenges and opportunities in generative modelling.*