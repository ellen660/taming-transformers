import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from my_code import BreathingPaths
from my_code import EEGPaths

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print(f'here {i}')
        example = self.data[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        # breakpoint()
        # print(f'here, {training_images_list_file}')
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = BreathingPaths(paths=paths, size=size)

class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = BreathingPaths(paths=paths, size=size)


