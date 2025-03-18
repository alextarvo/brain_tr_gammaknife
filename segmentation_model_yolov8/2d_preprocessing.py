import os
import nrrd
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import random
import shutil

def save_slices(mri_data, seg_data, orientation, patient_images_folder, patient_labels_folder, prefix):
    orientations = {
        "sagittal": (0, 1, 2),
        "axial": (1, 0, 2),
        "coronal": (2, 0, 1)
    }
    
    axis, height, width = orientations[orientation]
    num_slices = seg_data.shape[axis]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mri_data_torch = torch.tensor(mri_data, dtype=torch.float32, device=device)
    seg_data_torch = torch.tensor(seg_data, dtype=torch.uint8, device=device)
    
    for i in range(num_slices):
        mask = seg_data_torch.select(axis, i).cpu().numpy().astype(np.uint8)
        
        if np.count_nonzero(mask) == 0:
            continue  
        
        img = mri_data_torch.select(axis, i).cpu().numpy()
        
        if np.max(img) == np.min(img):
            img_norm = np.zeros_like(img, dtype=np.uint8)
        else:
            img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        
        image_filename = os.path.join(patient_images_folder, f"{prefix}_{orientation}_slice_{i}.png")
        cv2.imwrite(image_filename, img_norm)
        
        label_filename = os.path.join(patient_labels_folder, f"{prefix}_{orientation}_slice_{i}.txt")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        with open(label_filename, 'w') as f:
            for contour in contours:
                if len(contour) == 0:
                    continue
                normalized_contour = contour.reshape(-1, 2) / np.array([mask.shape[1], mask.shape[0]])
                flattened = normalized_contour.flatten()
                if len(flattened) < 6:
                    continue
                label_str = f"0 {' '.join(map(str, flattened))}\n"
                f.write(label_str)

def process_patient(patient_dir, images_root, labels_root):
    patient_id = os.path.basename(patient_dir)
    print(f"Processing patient: {patient_id}")
    
    mr_files = [f for f in os.listdir(patient_dir) if "_MR" in f and f.lower().endswith(".nrrd")]
    if not mr_files:
        print(f"Warning: No MR file found in {patient_dir}")
        return
    
    mr_path = os.path.join(patient_dir, mr_files[0])
    
    tumor_files = [f for f in os.listdir(patient_dir) if "_L" in f and f.lower().endswith(".nrrd")]
    if not tumor_files:
        print(f"Warning: No tumor segmentation files found in {patient_dir}")
        return
    
    mri_data, _ = nrrd.read(mr_path)
    
    for idx, tumor_filename in enumerate(tumor_files):
        tumor_path = os.path.join(patient_dir, tumor_filename)
        seg_data, _ = nrrd.read(tumor_path)
        
        prefix = f"{patient_id}_tumor{idx+1}"
        
        for orientation in ["sagittal", "axial", "coronal"]:
            patient_images_folder = os.path.join(images_root, orientation, prefix)
            patient_labels_folder = os.path.join(labels_root, orientation, prefix)
            os.makedirs(patient_images_folder, exist_ok=True)
            os.makedirs(patient_labels_folder, exist_ok=True)
            save_slices(mri_data, seg_data, orientation, patient_images_folder, patient_labels_folder, prefix)

def process_all_patients(root_dir, images_output, labels_output):
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)
    
    patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))]
    
    for patient_dir in patient_dirs:
        process_patient(patient_dir, images_output, labels_output)
        
def visualize_saved_slice(image_path, label_path, visual_file_output):
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found.")
        return
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    
    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} not found.")
    else:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                coords = list(map(float, parts[1:]))
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * width)
                    y = int(coords[i+1] * height)
                    points.append([x, y])
                points = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_rgb, [points], isClosed=True, color=(255, 0, 0), thickness=2)
    
    image_filename = os.path.join(visual_file_output, f"1_visual.png")
    cv2.imwrite(image_filename, img_rgb)
    
    # plt.figure(figsize=(6,6))
    # plt.imshow(img_rgb)
    # plt.title(f"Visualization: {os.path.basename(image_path)}")
    # plt.axis("off")
    # plt.show()
    
def allocate_test_data(train_images_root, train_labels_root, 
                       test_images_root, test_labels_root, test_split_ratio=0.1):
    """
    Allocates a portion of the training data to the test set by moving entire patient folders.
    """
    for orientation in ["sagittal", "axial", "coronal"]:
        train_img_orient = os.path.join(train_images_root, orientation)
        train_lbl_orient = os.path.join(train_labels_root, orientation)
        test_img_orient = os.path.join(test_images_root, orientation)
        test_lbl_orient = os.path.join(test_labels_root, orientation)
        
        os.makedirs(test_img_orient, exist_ok=True)
        os.makedirs(test_lbl_orient, exist_ok=True)
        
        patient_folders = [d for d in os.listdir(train_img_orient) if os.path.isdir(os.path.join(train_img_orient, d))]
        
        for folder in patient_folders:
            if random.random() < test_split_ratio:
                src_img = os.path.join(train_img_orient, folder)
                src_lbl = os.path.join(train_lbl_orient, folder)
                dst_img = os.path.join(test_img_orient, folder)
                dst_lbl = os.path.join(test_lbl_orient, folder)
                print(f"Allocating {folder} from {orientation} to test set.")
                shutil.move(src_img, dst_img)
                shutil.move(src_lbl, dst_lbl)
    
def allocate_validation_data(train_images_root, train_labels_root, val_images_root, val_labels_root, split_ratio=0.2):
    """
    Allocates a portion of the training data to validation by moving
    entire patient folders from the training directories to the validation directories.
    
    Parameters:
      - train_images_root: Root directory for training images (e.g., .../data/images/train)
      - train_labels_root: Root directory for training labels (e.g., .../data/labels/train)
      - val_images_root: Destination root for validation images (e.g., .../data/images/val)
      - val_labels_root: Destination root for validation labels (e.g., .../data/labels/val)
      - split_ratio: Fraction of patient folders to move to validation.
    """
    for orientation in ["sagittal", "axial", "coronal"]:
        train_img_orient = os.path.join(train_images_root, orientation)
        train_lbl_orient = os.path.join(train_labels_root, orientation)
        val_img_orient = os.path.join(val_images_root, orientation)
        val_lbl_orient = os.path.join(val_labels_root, orientation)
        
        os.makedirs(val_img_orient, exist_ok=True)
        os.makedirs(val_lbl_orient, exist_ok=True)
        
        # List all patient folders in the training images directory for the current orientation.
        patient_folders = [d for d in os.listdir(train_img_orient) if os.path.isdir(os.path.join(train_img_orient, d))]
        
        for folder in patient_folders:
            if random.random() < split_ratio:
                src_img = os.path.join(train_img_orient, folder)
                src_lbl = os.path.join(train_lbl_orient, folder)
                dst_img = os.path.join(val_img_orient, folder)
                dst_lbl = os.path.join(val_lbl_orient, folder)
                print(f"Allocating {folder} from {orientation} to validation.")
                shutil.move(src_img, dst_img)
                shutil.move(src_lbl, dst_lbl)

if __name__ == "__main__":
    root_dir = "EE577_Final_Project_Files/Brain-TR-GammaKnife-processed"
    images_output = "EE577_Final_Project_Files/2d_training_files/data/images/train"

    labels_output = "EE577_Final_Project_Files/2d_training_files/data/labels/train"
    
    process_all_patients(root_dir, images_output, labels_output)

    # Example visualization for a specific patient and slice 
    sample_image_path = os.path.join("EE577_Final_Project_Files/2d_training_files/data/images/train/axial/GK.103_1_tumor2/GK.103_1_tumor2_axial_slice_103.png")
    sample_label_path = os.path.join("EE577_Final_Project_Files/2d_training_files/data/labels/train/axial/GK.103_1_tumor2/GK.103_1_tumor2_axial_slice_103.png")
    
    visual_file_output = "EE577_Final_Project_Files/2d_training_files/label_overlay_output"
    
    # visualize_saved_slice(sample_image_path, sample_label_path, visual_file_output)
    
    # Define test output paths.
    test_images_output = "EE577_Final_Project_Files/2d_training_files/data/images/test"
    test_labels_output = "EE577_Final_Project_Files/2d_training_files/data/labels/test"
    
    # Allocate a percentage (10%) of the training data to the test set.
    allocate_test_data(images_output, labels_output, test_images_output, test_labels_output, test_split_ratio=0.1)
    
    # Define validation output paths.
    val_images_output = "EE577_Final_Project_Files/2d_training_files/data/images/val"
    val_labels_output = "EE577_Final_Project_Files/2d_training_files/data/labels/val"
    
    # Allocate a percentage (20%) of the training data to validation.
    allocate_validation_data(images_output, labels_output, val_images_output, val_labels_output, split_ratio=0.2)