import nrrd
import numpy as np
import cv2
import os
import torch
from ultralytics import YOLO

def run_inference_on_slice(model, slice_img):
    """
    Runs segmentation inference on a single slice.
    If the slice is grayscale, it is converted to a 3-channel image.
    Returns a binary segmentation mask (uint8, 0 or 255) resized to match the input slice.
    """
    original_shape = slice_img.shape[:2]
    # Convert grayscale to 3-channel if necessary
    if len(slice_img.shape) == 2:
        slice_img_color = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
    else:
        slice_img_color = slice_img

    # Run inference; the model accepts an image (numpy array)
    result = model(slice_img_color)[0]
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # shape: (n_objects, H, W)
        combined_mask = np.max(masks, axis=0)
        combined_mask = (combined_mask > 0.5).astype(np.uint8) * 255
    else:
        combined_mask = np.zeros(original_shape, dtype=np.uint8)

    # Resize the mask back to the original slice dimensions
    if combined_mask.shape != original_shape:
        combined_mask = cv2.resize(combined_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return combined_mask

def run_inference_on_nrrd(nrrd_path, seg_model_path, orientation='axial'):
    """
    Loads the NRRD volume, runs inference on each slice in the specified orientation,
    and returns a 3D volume of segmentation masks along with the original header.
    
    Orientation options:
      - 'axial': slices are taken as data[i, :, :] (iterates over the first dimension)
      - 'coronal': slices are taken as data[:, i, :] (iterates over the second dimension)
      - 'sagittal': slices are taken as data[:, :, i] (iterates over the third dimension)
    """
    # Load the original volume and header
    data, header = nrrd.read(nrrd_path)
    inference_dict = {}

    # Initialize the YOLO segmentation model and move it to the proper device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLO(seg_model_path)
    model.to(device)

    if orientation == 'axial':
        num_slices = data.shape[0]
        for i in range(num_slices):
            slice_img = data[i, :, :]
            if slice_img.dtype != np.uint8:
                slice_img = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mask = run_inference_on_slice(model, slice_img)
            inference_dict[i] = mask
        seg_volume = np.stack([inference_dict[i] for i in range(num_slices)], axis=0)

    elif orientation == 'coronal':
        num_slices = data.shape[1]
        for i in range(num_slices):
            slice_img = data[:, i, :]
            if slice_img.dtype != np.uint8:
                slice_img = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mask = run_inference_on_slice(model, slice_img)
            inference_dict[i] = mask
        temp_vol = np.stack([inference_dict[i] for i in range(num_slices)], axis=0)
        seg_volume = np.transpose(temp_vol, (1, 0, 2))

    elif orientation == 'sagittal':
        num_slices = data.shape[2]
        for i in range(num_slices):
            slice_img = data[:, :, i]
            if slice_img.dtype != np.uint8:
                slice_img = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mask = run_inference_on_slice(model, slice_img)
            inference_dict[i] = mask
        temp_vol = np.stack([inference_dict[i] for i in range(num_slices)], axis=0)
        seg_volume = np.transpose(temp_vol, (1, 2, 0))
    else:
        raise ValueError("Orientation must be 'axial', 'coronal', or 'sagittal'.")

    if seg_volume.shape != data.shape:
        print(f"Warning: Segmentation volume shape {seg_volume.shape} does not match original volume shape {data.shape}.")

    return seg_volume, header

def main():
    # File paths
    # path_to_mri_volume = "EE577_Final_Project_Files/Brain-TR-GammaKnife-processed/GK.199_1/GK.199_1_MR_t1.nrrd"
    # path_to_mri_volume = "EE577_Final_Project_Files/Brain-TR-GammaKnife-processed/GK.103_1/GK.103_1_MR_t1.nrrd"
    path_to_mri_volume = "EE577_Final_Project_Files/Brain-TR-GammaKnife-processed/GK.147_2/GK.147_2_MR_t1.nrrd"
    seg_model_path = "EE577_Final_Project_Files/2d_training_files/runs/train/weights/best.pt"
    output_nrrd_path = "EE577_Final_Project_Files/3d_seg_volume_result/reconstructed_tumor_segmentation.nrrd"
    
    # Set the orientation for slicing (axial, coronal, or sagittal)
    orientation = "axial"
    
    # Run segmentation inference on the NRRD volume.
    seg_volume, header = run_inference_on_nrrd(path_to_mri_volume, seg_model_path, orientation)
    
    # Update header dimension to match the segmentation volume
    header['dimension'] = seg_volume.ndim
    nrrd.write(output_nrrd_path, seg_volume.astype(np.uint8), header)
    print(f"Segmentation volume saved to: {output_nrrd_path}")

if __name__ == "__main__":
    main()
