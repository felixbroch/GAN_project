import torch
from torch import nn
import numpy as np
import math


class GAN_generator(nn.Module):

    def __init__(self):
        super.__init__()

    def forward(self, X):

        return X