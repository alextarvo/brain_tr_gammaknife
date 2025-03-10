import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.v2 as T

import numpy as np
import random


class CenterPadTransform:
    def __init__(self, target_size, do_random_shift=False):
        """
        Args:
            target_size (tuple): Desired output size (H, W).
        """
        self.target_size = target_size
        self.do_random_shift = do_random_shift

    def __call__(self, input_tensor):
        input_shape = np.array(input_tensor.shape)  # (D, H, W)
        interpolate_factor = np.max(input_tensor.shape / np.array(self.target_size))
        if interpolate_factor > 1.0:
            # print(f'Performing interpolation of the input image size {input_shape}; factor {interpolate_factor}')
            input_tensor = F.interpolate(input_tensor.unsqueeze(0).unsqueeze(0), scale_factor=1 / interpolate_factor,
                                         mode='trilinear')
            # print(f'Size of the interpolated tensor: {input_tensor.shape}')
            input_tensor = input_tensor.squeeze(0).squeeze(0)
            input_shape = np.array(input_tensor.shape)

        target_tensor = torch.zeros(self.target_size, dtype=input_tensor.dtype, device=input_tensor.device)

        # print(f'target tensor shape: {target_tensor.shape}, input tensor shape: {input_shape}')
        if self.do_random_shift:
            # This should be used for training. Shift the patch randomly within the (self.target_size) boundaries
            start_d = random.randint(0, (self.target_size[0] - input_shape[0]))
            start_h = random.randint(0, (self.target_size[1] - input_shape[1]))
            start_w = random.randint(0, (self.target_size[2] - input_shape[2]))
        else:
            # Compute the starting indices for centering
            start_d = (self.target_size[0] - input_shape[0]) // 2
            start_h = (self.target_size[1] - input_shape[1]) // 2
            start_w = (self.target_size[2] - input_shape[2]) // 2

        # Place the input tensor in the center
        target_tensor[
        start_d:start_d + input_shape[0],
        start_h:start_h + input_shape[1],
        start_w:start_w + input_shape[2]] = input_tensor

        return target_tensor
