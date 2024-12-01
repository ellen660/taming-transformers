import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from my_code.utils import detect_motion_iterative, signal_crop, norm_sig
from scipy.ndimage import zoom
import torch
from my_code.spectrogram import recompute_breathing_rate

class BreathingPaths(Dataset):
    def __init__(self, paths, size=None, labels=None):
        self.size = size
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
    
    def __len__(self):
        return self._length

    def preprocess_image(self, breathing_path):
        # print(f'image_path: {image_path}')
        breathing, fs = np.load(breathing_path)['data'], np.load(breathing_path)['fs']

        signal, _, _ = detect_motion_iterative(breathing, fs)
        signal = signal_crop(signal)
        signal = norm_sig(signal)

        # if fs != 10:
        #     signal = zoom(signal, 10/fs)
        #     fs = 8
        if signal.shape[0] >= self.size:
            feature = signal[:self.size]
        else:
            feature = np.pad(signal, (0, self.size - signal.shape[0]))

        feature = np.expand_dims(feature, axis=0)
        #turn feature into tensor
        feature = torch.tensor(feature, dtype=torch.float32)
        
        return feature
    
    def preprocess_breathing(self, breathing_path):
        breathing, fs = np.load(breathing_path)['data'], np.load(breathing_path)['fs']
        breathing = self.preprocess_image(breathing_path)
        
        bpm, _, spec = recompute_breathing_rate(
            breathing,
            spec_step_sec=5,
            spec_win_sec=30,
            npad=20,
            filter_half_window_size=10,
            fs=fs,
            cutoff_bpm=40,
        )

        # print(f'spec shape: {spec.shape}')
        #Here we cut the data from 0 to 60 bpm to 8 to 40 bpm
        # the range of spec is 400 so if less than 400 or more than 2400 we set it to NaN
        # specifically, spec is 2d, spec[0] is the frequency, spec[1] is the time
        # print(f'spec shape: {spec.shape}')
        spec = spec[80:, :]
        spec = spec[:, :320]
        assert spec.shape[0] == 320
        assert spec.shape[1] == 320

        # breakpoint()
        return spec

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_breathing(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
