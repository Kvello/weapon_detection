import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np

class ImageDataset(datasets.ImageFolder):
    """
    Dataset class for loading images for classification
    The class preloades all images, thus the memory usage is high
    """
    def __init__(self, data_dir, transform=None):
        super(ImageDataset, self).__init__(data_dir, transform)