from radiomics import featureextractor
import numpy as np
from fmcib.models import fmcib_model
import torch
import torch.nn.functional as F

class RadiomicsExtractor(object):
    def __init__(self):
        pass


class PyRadiomicsExtractor(RadiomicsExtractor):
    def __init__(self):
        super(PyRadiomicsExtractor, self).__init__()
        self.extractor = featureextractor.RadiomicsFeatureExtractor()

    def extract_features(self, **kwargs):
        mri_path = kwargs.get('mri_path', None)  # Returns None if 'mri' is not provided
        tumor_mask_path = kwargs.get('tumor_mask_path', None)

        if mri_path is None or tumor_mask_path is None:
            raise ValueError("Both 'mri' and 'tumor_mask' must be provided")

        features = self.extractor.execute(mri_path, tumor_mask_path)
        # for key, value in features.items():
        #     print(f"{key}: {value}")
        filtered_metrics = {k.replace("original_", ""): v for k, v in features.items() if k.startswith("original_")}

        for key in filtered_metrics.keys():
            value = filtered_metrics[key]
            if (isinstance(value, np.ndarray) and value.shape == ()) or (
                    isinstance(value, list) and len(value) == 1):
                value = float(value)
                filtered_metrics[key] = value
        return filtered_metrics


class FCIBImageExtractor(RadiomicsExtractor):
    def __init__(self):
        super(FCIBImageExtractor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fmcib_model().to(self.device)

        self.target_tensor_size = 50

    def center_pad_tensor(self, input_tensor, target_size):
        """
        Centers a smaller 3D tensor into a larger 3D tensor of size (50,50,50).
        Assumes input_tensor is smaller than target_size.

        Args:
            input_tensor (torch.Tensor): A 3D tensor of shape (D, H, W)
            target_size (tuple): Target shape (default: 50,50,50)

        Returns:
            torch.Tensor: Padded tensor of shape (50,50,50)
        """
        input_shape = input_tensor.shape  # (D, H, W)
        target_tensor = torch.zeros(target_size, dtype=input_tensor.dtype, device=input_tensor.device)

        # Compute the starting indices for centering
        start_d = (target_size[0] - input_shape[0]) // 2
        start_h = (target_size[1] - input_shape[1]) // 2
        start_w = (target_size[2] - input_shape[2]) // 2

        # Place the input tensor in the center
        target_tensor[
        start_d:start_d + input_shape[0],
        start_h:start_h + input_shape[1],
        start_w:start_w + input_shape[2]] = input_tensor

        return target_tensor

    def extract_features(self, **kwargs):
        image_patch = kwargs.get('image_patch', None)  # Returns None if 'mri' is not provided
        image_patch_tensor = torch.tensor(image_patch, dtype=torch.float32)
        target_size = (self.target_tensor_size, self.target_tensor_size,self.target_tensor_size)
        padded_tensor = self.center_pad_tensor(image_patch_tensor, target_size=target_size)
        tensor_in = padded_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        metrics = self.model(tensor_in).detach().cpu().squeeze().numpy()
        assert len(metrics.shape) == 1
        nn_metrics = {f'feature_{i}': metrics[i] for i in range(metrics.shape[0])}
        return nn_metrics
