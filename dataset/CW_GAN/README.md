# Training Outputs Directory

This directory contains the outputs generated during GAN training.

## Expected Contents:
```
CW_GAN/
├── real_samples.png             # Sample real images from dataset
├── fake_samples_epoch_*.png     # Generated images by epoch and hyperparameters
├── GAN_G_model.pth             # Saved generator model
├── GAN_D_model.pth             # Saved discriminator model
└── loss_data.pkl               # Training loss history
```

## File Naming Convention:
- Generated images: `fake_samples_epoch_{epoch:03d}_lrG_{lr_g:.6f}_lrD_{lr_d:.6f}_{timestamp}.png`
- Models are saved after training completion
- Loss data is pickled for later analysis

**Note**: The actual output files (.png, .pth, .pkl) are excluded from Git due to their size, but this directory structure will be preserved.
