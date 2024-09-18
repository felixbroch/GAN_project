import torch
from torch import nn
import numpy as np
import math
from torch.utils.data import Dataset

class GAN_dataset(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, index) :
        return 0