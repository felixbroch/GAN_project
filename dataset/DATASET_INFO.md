# Dataset Information

## Pistachio Image Dataset

This directory contains the dataset used for GAN training, which is excluded from version control due to file size constraints.

### Dataset Details
- **Source**: [Kaggle Pistachio Image Dataset](https://www.kaggle.com/datasets/muratkoklu/pistachio-image-dataset)
- **Variety**: Kirmizi Pistachio (red pistachio variety)
- **Format**: JPG images with standardized naming convention
- **Size**: Large image files (not suitable for Git repositories)

### Directory Structure
```
dataset/
├── DATASET_INFO.md              # This file
├── Pistachio_Image_Dataset/     # Original dataset directory (structure tracked)
│   ├── README.md                # Dataset setup instructions
│   ├── Kirmizi_Pistachio/       # Red pistachio images (*.jpg files excluded from Git)
│   └── Pistachio_Image_Dataset_Request.txt
└── CW_GAN/                      # Training outputs directory (structure tracked)
    ├── README.md                # Output directory information
    ├── real_samples.png         # Sample real images (excluded from Git)
    ├── fake_samples_*.png       # Generated images (excluded from Git)
    └── model checkpoints        # Model files (excluded from Git)
```

### Setup Instructions

To use this project, you need to download the dataset separately:

1. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/muratkoklu/pistachio-image-dataset)
2. **Extract** the dataset to `dataset/Pistachio_Image_Dataset/`
3. **Verify** that images are in `dataset/Pistachio_Image_Dataset/Kirmizi_Pistachio/`
4. **Run the main notebook** from the root directory: `GAN_dataset_new copy.ipynb`

### Why Dataset Files Are Excluded

Large dataset files are excluded from Git version control because:
- **Large file sizes** exceed GitHub's recommended limits
- **Binary files** don't benefit from version control
- **Reproducibility** is achieved through download instructions and preprocessing code
- **Best practice** for machine learning projects

**Note**: The directory structure and documentation files are tracked in Git, but the actual image files (.jpg, .png) and model files (.pth, .pkl) are excluded.

### Alternative: Git LFS

For projects requiring dataset version control, consider using Git Large File Storage (LFS) for large binary files.
