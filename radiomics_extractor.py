from radiomics import featureextractor
import numpy as np
from fmcib.models import fmcib_model
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T

from extractor_transform import CenterPadTransform
from fcib_uptune import FCIBUptune, FCIB_OUTPUT_SIZE, NUM_CLASSES


class RadiomicsExtractor(object):
    def __init__(self):
        pass

    def extract_features(self, mri_path=None, tumor_mask_path=None, image_patch=None):
        pass

    def show_slice(arr_slice, mask=None):
        plt.figure(figsize=(6, 6))
        plt.imshow(arr_slice, cmap="gray")
        if mask is not None:
            plt.imshow(mask, cmap="Blues", alpha=0.3)
        plt.axis("off")
        plt.show()




class PyRadiomicsExtractor(RadiomicsExtractor):
    def __init__(self):
        super(PyRadiomicsExtractor, self).__init__()
        self.extractor = featureextractor.RadiomicsFeatureExtractor()

    def extract_features(self, mri_path=None, tumor_mask_path=None, image_patch = None):
        # mri_path = kwargs.get('mri_path', None)  # Returns None if 'mri' is not provided
        # tumor_mask_path = kwargs.get('tumor_mask_path', None)

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
        print(f'Running FCBI extractor model on device {self.device}')
        self.target_tensor_size = 50
        self.transform = T.Compose([ CenterPadTransform(target_size=(
            self.target_tensor_size, self.target_tensor_size, self.target_tensor_size), do_random_shift=True),
        ])

    def extract_features(self, mri_path=None, tumor_mask_path=None, image_patch=None):
        # image_patch = kwargs.get('image_patch', None)  # Returns None if 'mri' is not provided
        if image_patch is None:
            raise ValueError("image_patch must be provided")
        image_patch_tensor = torch.tensor(image_patch, dtype=torch.float32)
        target_size = (self.target_tensor_size, self.target_tensor_size, self.target_tensor_size)
        padded_tensor = self.transform(image_patch_tensor)
        tensor_in = padded_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        metrics = self.model(tensor_in).detach().cpu().squeeze().numpy()
        assert len(metrics.shape) == 1
        nn_metrics = {f'feature_{i}': metrics[i] for i in range(metrics.shape[0])}
        return nn_metrics


class FCIBTunedImageExtractor(RadiomicsExtractor):
    def __init__(self, saved_model_path):
        super(FCIBTunedImageExtractor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fcib_config = {'dropout_prob': 0.3,
              'fcib_output_size': FCIB_OUTPUT_SIZE,
              'head_interim_size': 30,
              'num_classes': NUM_CLASSES,
              'fine_tune_mode': 'full',
              }

        self.model = FCIBUptune(fcib_config)
        saved_model = torch.load(saved_model_path)
        self.model.load_state_dict(saved_model)
        self.model.to(self.device)

        print(f'Running FCBI fine-tuned extractor model on device {self.device}')
        self.target_tensor_size = 50
        self.transform = T.Compose([
            CenterPadTransform(target_size=(
                self.target_tensor_size, self.target_tensor_size, self.target_tensor_size), do_random_shift=True),
            ])

    def extract_features(self, mri_path=None, tumor_mask_path=None, image_patch=None):
        # image_patch = kwargs.get('image_patch', None)  # Returns None if 'mri' is not provided
        if image_patch is None:
            raise ValueError("image_patch must be provided")
        image_patch_tensor = torch.tensor(image_patch, dtype=torch.float32)
        padded_tensor = self.transform(image_patch_tensor)
        tensor_in = padded_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        _, metrics_t = self.model(tensor_in)
        metrics = metrics_t.detach().cpu().squeeze().numpy()
        assert len(metrics.shape) == 1
        nn_metrics = {f'feature_{i}': metrics[i] for i in range(metrics.shape[0])}
        return nn_metrics
