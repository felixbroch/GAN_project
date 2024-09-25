import torch
from torch import nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
import torch.nn.functional as F



device = torch.device("cpu")

# Definition of hyperparameters
num_epochs = 100
learning_rate_g = 0.0001
learning_rate_d = 0.0002
latent_vector_size = 40

# Other hyperparams
channel_input = 32
channel_output = 3




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define the generator's layers with explicit parameter naming
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=latent_vector_size, out_channels=channel_input * 16, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=channel_input * 16)
        self.relu1 = nn.LeakyReLU(inplace=True)
        
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=channel_input * 16, out_channels=channel_input * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=channel_input * 8)
        self.relu2 = nn.LeakyReLU(inplace=True)
        
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=channel_input * 8, out_channels=channel_input * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=channel_input * 4)
        self.relu3 = nn.LeakyReLU(inplace=True)
        
        self.conv_transpose4 = nn.ConvTranspose2d(in_channels=channel_input * 4, out_channels= channel_output, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, z):
        z = self.relu1(self.batch_norm1(self.conv_transpose1(z)))
        z = self.relu2(self.batch_norm2(self.conv_transpose2(z)))
        z = self.relu3(self.batch_norm3(self.conv_transpose3(z)))
        z = self.tanh(self.conv_transpose4(z))
        return z

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(channel_output, channel_input*4, 4, 2, 1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(channel_input*4)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(channel_input*4, channel_input*8, 4, 2, 1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(channel_input*8)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(channel_input*8, channel_input * 16, 4, 2, 1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(channel_input*16)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(channel_input*16, channel_input * 32, 4, 2, 1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(channel_input*32)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv2d(channel_input*32, channel_input * 64, 2, 1, 1, bias=False)
        self.batch_norm5 = nn.BatchNorm2d(channel_input*64)
        self.leaky_relu5 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv6 = nn.Conv2d(channel_input*64, 1, 3, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu1(self.batch_norm1(self.conv1(x)))
        x = self.leaky_relu2(self.batch_norm2(self.conv2(x)))
        x = self.leaky_relu3(self.batch_norm3(self.conv3(x)))
        x = self.leaky_relu4(self.batch_norm4(self.conv4(x)))
        x = self.leaky_relu5(self.batch_norm5(self.conv5(x)))
        x = self.sigmoid(self.conv6(x))
        return x


# custom weights initialisation called on netG and netD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

use_weights_init = True

model_G = Generator().to(device)
if use_weights_init:
    model_G.apply(weights_init)
params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
print("Total number of parameters in Generator is: {}".format(params_G))
print(model_G)
print('\n')

model_D = Discriminator().to(device)
if use_weights_init:
    model_D.apply(weights_init)
params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
print("Total number of parameters in Discriminator is: {}".format(params_D))
print(model_D)
print('\n')

print("Total number of parameters is: {}".format(params_G + params_D))


# Define a loss function
def loss_function(out, real_or_fake):
    if real_or_fake == 'real':
        loss = F.binary_cross_entropy(out, torch.ones(out.size()).to(device))
    elif real_or_fake == 'fake':
        loss = F.binary_cross_entropy(out, torch.zeros(out.size()).to(device))
    else:
        raise ValueError('real_or_fake must be either "real" or "fake"')
    return loss


# Kept the code to modify the dataset, have not run it yet to keep the original data, will need to be run only once

transform = transforms.Compose([
    transforms.Resize((128, 128)), # Dimensions of the picture, trade-off between quality and computational cost
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

def preprocess_image(image_path):
    image = Image.open(image_path)
    return transform(image)



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


