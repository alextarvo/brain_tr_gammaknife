import os
import nrrd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Saves MRI slices as PNG images.
def save_mri_slices(mri_nrrd_path, output_folder, prefix="mri_slice"):
    os.makedirs(output_folder, exist_ok=True)
    mri_data, _ = nrrd.read(mri_nrrd_path)
    num_slices = mri_data.shape[0]
    
    for i in range(num_slices):
        img = mri_data[i, :, :]
        # Prevent division-by-zero if image has constant values
        if np.max(img) == np.min(img):
            img_norm = np.zeros_like(img, dtype=np.uint8)
        else:
            img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        filename = os.path.join(output_folder, f"{prefix}_slice_{i}.png")
        cv2.imwrite(filename, img_norm)
    print(f"Saved {num_slices} MRI slices to {output_folder}")
    return num_slices

# Converts a tumor segmentation NRRD file to YOLO segmentation labels.
# Each slice is saved as a separate .txt file.
def convert_nrrd_to_yolo(nrrd_path, output_folder, prefix="mri_slice"):
    os.makedirs(output_folder, exist_ok=True)
    segmentation_data, _ = nrrd.read(nrrd_path)
    num_slices = segmentation_data.shape[0]

    for slice_idx in range(num_slices):
        mask = segmentation_data[slice_idx, :, :].astype(np.uint8)
        label_filename = os.path.join(output_folder, f"{prefix}_slice_{slice_idx}.txt")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        with open(label_filename, 'w') as f:
            for contour in contours:
                if len(contour) == 0:
                    continue
                # Normalize coordinates to [0, 1]
                normalized_contour = contour.reshape(-1, 2) / np.array([mask.shape[1], mask.shape[0]])
                flattened = normalized_contour.flatten()
                # Only write if we have at least 3 points (i.e. 6 numbers)
                # Mitigates issue with segmentation being mistook for detection (boxes).
                if len(flattened) < 6:
                    continue
                label_str = f"0 {' '.join(map(str, flattened))}\n"
                f.write(label_str)
        print(f"Saved YOLO label: {label_filename}")

# Processes one patient directory by locating the MR scan and tumor scans.
# For each tumor scan, the MR images are duplicated and the tumor segmentation is converted to YOLO labels.
# Mitigates issue of too many tumor segmentation labels per one MR image.
def process_patient(patient_dir, images_root, labels_root):
    patient_id = os.path.basename(patient_dir)
    print(f"Processing patient: {patient_id}")

    # Identify MR scan (file containing "_MR" in its name)
    mr_files = [f for f in os.listdir(patient_dir) if "_MR" in f and f.lower().endswith(".nrrd")]
    if not mr_files:
        print(f"Warning: No MR file found in {patient_dir}")
        return
    mr_path = os.path.join(patient_dir, mr_files[0])
    
    # Identify tumor scans (files containing "_L" in their name)
    tumor_files = [f for f in os.listdir(patient_dir) if "_L" in f and f.lower().endswith(".nrrd")]
    if not tumor_files:
        print(f"Warning: No tumor segmentation files found in {patient_dir}")
        return

    # For each tumor scan, create an images and labels subfolder.
    for idx, tumor_filename in enumerate(tumor_files):
        tumor_path = os.path.join(patient_dir, tumor_filename)
        # Define a unique prefix for this patient-tumor pair.
        prefix = f"{patient_id}_tumor{idx+1}"
        
        # Create output directories for images and labels.
        patient_images_folder = os.path.join(images_root, prefix)
        patient_labels_folder = os.path.join(labels_root, prefix)
        
        # Duplicate MR scan images (training images)
        num_slices = save_mri_slices(mr_path, patient_images_folder, prefix=prefix)
        # Convert tumor segmentation to YOLO compatible labels.
        convert_nrrd_to_yolo(tumor_path, patient_labels_folder, prefix=prefix)

# Iterates through the patient directories in the root directory.
def process_all_patients(root_dir, images_output, labels_output):
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)
    
    patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))]
    
    for patient_dir in patient_dirs:
        process_patient(patient_dir, images_output, labels_output)

# Visualizes YOLO segmentation labels by overlaying onto MRI slices.
def visualize_labels(image_folder, label_folder, slice_indices):
    for slice_idx in slice_indices:
        image_path = os.path.join(image_folder, f"{os.path.basename(image_folder)}_slice_{slice_idx}.png")
        label_path = os.path.join(label_folder, f"{os.path.basename(image_folder)}_slice_{slice_idx}.txt")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            continue
        
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
                    # parts[0] is class_id (should be 0)
                    coords = list(map(float, parts[1:]))
                    points = []
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] * width)
                        y = int(coords[i+1] * height)
                        points.append([x, y])
                    points = np.array(points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_rgb, [points], isClosed=True, color=(255, 0, 0), thickness=2)
                    
        plt.figure(figsize=(6,6))
        plt.imshow(img_rgb)
        plt.title(f"Slice {slice_idx} - {os.path.basename(image_folder)}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    # Overall root directory containing patient folders
    root_dir = "development/EE577_Final_Project_Files/Brain-TR-GammaKnife-processed"
    
    # Output directories for training images and labels
    images_output = "development/EE577_Final_Project_Files/data/images/train"
    labels_output = "development/EE577_Final_Project_Files/data/labels/train"
    
    # Process all patient directories
    process_all_patients(root_dir, images_output, labels_output)
    
    # Visualize example tumor/MR pair
    sample_patient_folder = os.path.join(images_output, "GK.103_1_tumor1")
    sample_label_folder = os.path.join(labels_output, "GK.103_1_tumor1")
    
    # Specify slice indices to visualize (adjust as needed)
    slice_indices = [163]
    
    visualize_labels(sample_patient_folder, sample_label_folder, slice_indices)
