import nrrd
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

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

def compute_dice(pred, gt):
    """
    Computes the Dice score between the prediction and ground truth binary masks.
    Assumes pred and gt are binary (0 and 1).
    """
    intersection = np.sum(pred * gt)
    denominator = np.sum(pred) + np.sum(gt)
    if denominator == 0:
        return 1.0  # Both are empty
    dice = (2 * intersection) / denominator
    return dice

def compute_ap(pred, gt):
    """
    Computes the Average Precision (AP) score using a precision-recall curve.
    Both pred and gt should be flattened binary arrays (0 and 1).
    """
    if np.sum(gt) == 0:
        if np.sum(pred) == 0:
            return 1.0
        else:
            return 0.0
    return average_precision_score(gt, pred)

def evaluate_volume_metrics(pred_volume, gt_volume):
    """
    Evaluates the metrics for the entire volume.
    Both volumes should be binary (0 and 1).
    Returns dice and AP scores.
    """
    pred_flat = pred_volume.flatten()
    gt_flat = gt_volume.flatten()
    dice = compute_dice(pred_flat, gt_flat)
    ap = compute_ap(pred_flat, gt_flat)
    return dice, ap

def evaluate_slice_metrics(pred_volume, gt_volume, orientation='axial'):
    """
    Evaluates metrics slice-by-slice along the given orientation.
    Returns lists of dice scores and AP scores for each slice, as well as the mean scores.
    """
    dice_scores = []
    ap_scores = []
    
    if orientation == 'axial':
        num_slices = pred_volume.shape[0]
        for i in range(num_slices):
            pred_slice = pred_volume[i, :, :].flatten()
            gt_slice = gt_volume[i, :, :].flatten()
            dice_scores.append(compute_dice(pred_slice, gt_slice))
            ap_scores.append(compute_ap(pred_slice, gt_slice))
            
    elif orientation == 'coronal':
        num_slices = pred_volume.shape[1]
        for i in range(num_slices):
            pred_slice = pred_volume[:, i, :].flatten()
            gt_slice = gt_volume[:, i, :].flatten()
            dice_scores.append(compute_dice(pred_slice, gt_slice))
            ap_scores.append(compute_ap(pred_slice, gt_slice))
            
    elif orientation == 'sagittal':
        num_slices = pred_volume.shape[2]
        for i in range(num_slices):
            pred_slice = pred_volume[:, :, i].flatten()
            gt_slice = gt_volume[:, :, i].flatten()
            dice_scores.append(compute_dice(pred_slice, gt_slice))
            ap_scores.append(compute_ap(pred_slice, gt_slice))
    else:
        raise ValueError("Orientation must be 'axial', 'coronal', or 'sagittal'.")
    
    mean_dice = np.mean(dice_scores)
    mean_ap = np.mean(ap_scores)
    return dice_scores, ap_scores, mean_dice, mean_ap

def main():
    # File paths
    # path_to_mri_volume = "EE577_Final_Project_Files/Brain-TR-GammaKnife-processed/GK.103_1/GK.103_1_MR_t1.nrrd"
    # tumor_gt_path = "EE577_Final_Project_Files/Brain-TR-GammaKnife-processed/GK.103_1/GK.103_1_LR atrium.nrrd"
    
    # path_to_mri_volume = "EE577_Final_Project_Files/Brain-TR-GammaKnife-processed/GK.224_2/GK.224_2_MR_t1.nrrd"
    # tumor_gt_path = "EE577_Final_Project_Files/Brain-TR-GammaKnife-processed/GK.224_2/GK.224_2_LRtCerebellar.nrrd"
    
    path_to_mri_volume = "EE577_Final_Project_Files/Brain-TR-GammaKnife-processed/GK.243_3/GK.243_3_MR_t1.nrrd"
    tumor_gt_path = "EE577_Final_Project_Files/Brain-TR-GammaKnife-processed/GK.243_3/GK.243_3_L_LtHippocampal.nrrd"
    
    seg_model_path = "EE577_Final_Project_Files/2d_training_files/runs/train2/weights/best.pt"
    output_nrrd_path = "EE577_Final_Project_Files/3d_seg_volume_result/reconstructed_tumor_segmentation.nrrd"
    
    # Set the orientation for slicing (axial, coronal, or sagittal)
    orientation = "axial"
    
    # Run segmentation inference on the MRI volume.
    pred_volume, header = run_inference_on_nrrd(path_to_mri_volume, seg_model_path, orientation)
    
    # Save the predicted segmentation volume as before.
    header['dimension'] = pred_volume.ndim
    nrrd.write(output_nrrd_path, pred_volume.astype(np.uint8), header)
    print(f"Segmentation volume saved to: {output_nrrd_path}")
    
    # Load the tumor ground truth volume.
    gt_volume, gt_header = nrrd.read(tumor_gt_path)
    
    # Make sure both volumes have the same shape.
    if pred_volume.shape != gt_volume.shape:
        print(f"Warning: Predicted volume shape {pred_volume.shape} and ground truth volume shape {gt_volume.shape} do not match.")
    
    # Convert both volumes to binary (0 and 1). Assuming tumor pixels are nonzero.
    pred_volume_bin = (pred_volume > 127).astype(np.uint8)
    gt_volume_bin = (gt_volume > 0).astype(np.uint8)
    
    # Compute volume-level metrics.
    vol_dice, vol_ap = evaluate_volume_metrics(pred_volume_bin, gt_volume_bin)
    print(f"\nVolume-level Dice Score: {vol_dice:.4f}")
    print(f"Volume-level mAP (AP): {vol_ap:.4f}")
    print("\n")
    
    # Compute slice-level metrics.
    slice_dices, slice_aps, mean_slice_dice, mean_slice_ap = evaluate_slice_metrics(pred_volume_bin, gt_volume_bin, orientation)
    print(f"Mean slice-level Dice Score: {mean_slice_dice:.4f}")
    print(f"Mean slice-level mAP (AP): {mean_slice_ap:.4f}\n")
    
if __name__ == "__main__":
    main()
