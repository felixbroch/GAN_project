import torch
from torch import nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os

class GAN_dataset(Dataset):

    def __init__(self, image_paths, transform = None):
        super().__init__()
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) :
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image


# Kept the code to modify the dataset, have not run it yet to keep the original data, will need to be run only once

# transform = transforms.Compose([
#     transforms.Resize((128, 128)), # Dimensions of the picture, trade-off between quality and computational cost
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
# ])

# def preprocess_image(image_path):
#     image = Image.open(image_path)
#     return transform(image)



# Define the directory and pattern to match the filenames
# Faut bien que dans ton terminal tu sois dans le folder 'French GAN' sinon ca l'accedera pas 
# C'est un peu chiant mais j'ai pas trouve mieux pour l'instant
image_directory = 'src/dataset/Pistachio_Image_Dataset/Kirmizi_Pistachio'

pattern = os.path.join(image_directory, 'kirmizi *.jpg')

# Use glob to get all the file paths that match the pattern
image_paths = glob.glob(pattern)
# print(len(image_paths))

# Loop through the matched image paths
for image_path in image_paths:
    print(image_path)

# Have not tested this part of the code yet want to make sure you're happy with the current code and set up
# dataset = GAN_dataset(image_paths, transform=transform)
# dataloader = DataLoader(dataset, batch_size=len(image_paths), shuffle=True)


