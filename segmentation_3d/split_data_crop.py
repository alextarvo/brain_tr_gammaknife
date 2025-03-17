# -*- encoding: utf-8 -*-
'''
file       :split_data.py
Description:1.split(train, val and test)
Author     :zhiwei tan
version    :python3.9.6
'''

'''
before
/Volumes/macdata/dataset_read/brain_mri/brain_mri/output/
    ├── GK.476_1/
        ├── MR_t1.nii.gz  
        ├── lesion1.nii.gz 
        ├── lesion2.nii.gz 
        ├── dose.nii.gz   
    ├── GK.487_1/
after
/Volumes/macdata/dataset_read/brain_mri/brain_mri/dataset_split/
    ├── train/
        ├── image/
        ├── label/
    ├── val/
    ├── test/


'''

# Re-load the dataset since the execution state was reset
import pandas as pd
import numpy as np
import os
import shutil
import nibabel as nib




def split_data(xlsx_path):
    xls = pd.ExcelFile(xlsx_path)

    # Load the lesion_level sheet
    lesion_level_df = pd.read_excel(xls, sheet_name='lesion_level')

    # Get all unique patient IDs
    unique_patient_ids = lesion_level_df['unique_pt_id'].unique() # 47
    # Set random seed for reproducibility
    np.random.seed(42)

    # Shuffle the patient IDs
    np.random.shuffle(unique_patient_ids)

    # Calculate the split sizes
    num_patients = len(unique_patient_ids)
    train_size = int(0.7 * num_patients)
    val_size = int(0.15 * num_patients)
    test_size = num_patients - train_size - val_size  # Ensure total count is correct

    # Split dataset
    train_patients = unique_patient_ids[:train_size]
    val_patients = unique_patient_ids[train_size:train_size + val_size]
    test_patients = unique_patient_ids[train_size + val_size:]

    # Create a dictionary to store the dataset split
    dataset_split = {
        'train': train_patients,
        'val': val_patients,
        'test': test_patients
    }

    # Display the number of patients in each split
    dataset_split_counts = {
        'train': len(train_patients),
        'val': len(val_patients),
        'test': len(test_patients)
    }
    return dataset_split


def create_split_folders(save_path):
    """Create train/val/test directories with image and label subdirectories."""

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(save_path, split, "image"), exist_ok=True)
        os.makedirs(os.path.join(save_path, split, "label"), exist_ok=True)

def copy_image_file(folder_path, target_dir, folder_name):
    """复制 MR_t1.nii.gz 影像文件到对应数据集的 image 文件夹。"""
    image_files = [f for f in os.listdir(folder_path) if f.endswith("_MR_t1.nii.gz")]

    if image_files:
        image_file_path = os.path.join(folder_path, image_files[0])  # 选取第一个匹配的文件
        shutil.copy(image_file_path, os.path.join(target_dir, f"{folder_name}_MR_t1.nii.gz"))


def merge_and_save_labels(folder_path, target_dir, folder_name):
    """
    Merge all qualified label (.nii.gz) files and save them to the label directory.
    Rule: exclude files ending with _dose.nii.gz and _MR_t1.nii.gz.
    """
    label_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith(".nii.gz") and not f.endswith("_dose.nii.gz") and not f.endswith("_MR_t1.nii.gz")
    ]

    if not label_files:
        print(folder_path)
        return  # 如果没有符合条件的 label 文件，则不处理

    combined_label = None
    for label_file in label_files:
        nii = nib.load(label_file)
        label_data = nii.get_fdata()

        # 合并 label（按最大值叠加）
        if combined_label is None:
            combined_label = label_data
        else:
            combined_label = np.maximum(combined_label, label_data)

    # 保存合并后的 label 文件
    if combined_label is not None:
        label_nii = nib.Nifti1Image(combined_label, affine=nii.affine)
        nib.save(label_nii, os.path.join(target_dir, f"{folder_name}_label.nii.gz"))

def process_dataset(ori_path,dataset_split,save_path):
    """Traverse the output directory and divide the images and labels according to train/val/test."""
    for folder_name in os.listdir(ori_path):
        folder_path = os.path.join(ori_path, folder_name)

        # 确保是文件夹，并检查是否包含 patient_id
        if os.path.isdir(folder_path):
            assigned_split = None

            for split, patient_list in dataset_split.items():
                # 只要 patient_id 作为字符串出现在 folder_name 中，就认为匹配
                for patient_id in patient_list:
                    if str(patient_id) in folder_name:
                        assigned_split = split
                        break
                if assigned_split:
                    break  # 该文件夹已分配到某个数据集

            if assigned_split is None:
                continue  # 该患者不属于 train/val/test，跳过

            # 获取目标文件夹
            image_target_dir = os.path.join(save_path, assigned_split, "image")
            label_target_dir = os.path.join(save_path, assigned_split, "label")

            # 处理影像数据
            copy_image_file(folder_path, image_target_dir, folder_name)

            # 处理 label 数据（合并并保存）
            merge_and_save_labels(folder_path, label_target_dir, folder_name)



def find_mr_t1_file(folder_path):
    """Find the `_MR_t1.nii.gz` image file in the given folder."""
    image_files = [f for f in os.listdir(folder_path) if f.endswith("_MR_t1.nii.gz")]
    return os.path.join(folder_path, image_files[0]) if image_files else None


def extract_z_slices(image_data, lesion_data):
    """
    Calculate the z-axis extent of the lesion and extract the corresponding slice.
    """
    z_indices = np.any(lesion_data, axis=(0, 1))  
    min_z, max_z = np.where(z_indices)[0][[0, -1]]  
    return image_data[:, :, min_z:max_z+1], min_z, max_z  


MIN_SLICES = 16  # Minimum number of z-axis slices


def adjust_z_range(min_z, max_z, total_slices):
    """
    If the z-axis range is smaller than MIN_SLICES, extend the range.
    - Prefer balanced expansion.
    - If one side reaches the boundary, expand the other side instead.
    """
    num_slices = max_z - min_z + 1  # Compute current number of slices
    if num_slices >= MIN_SLICES:
        return min_z, max_z  # Already meets the requirement

    extra_slices = MIN_SLICES - num_slices  # Number of additional slices needed
    add_before = extra_slices // 2  # Expand backward first
    add_after = extra_slices - add_before  # Expand forward afterward

    # If min_z is too small, shift expansion to the other side
    if min_z - add_before < 0:
        add_after += add_before - min_z  # Transfer the backward expansion to forward
        add_before = min_z

    # If max_z is too large, shift expansion to the other side
    if max_z + add_after >= total_slices:
        add_before += (max_z + add_after) - (total_slices - 1)  # Transfer overflow to backward
        add_after = (total_slices - 1) - max_z

    # Compute the final min_z and max_z
    min_z = max(0, min_z - add_before)
    max_z = min(total_slices - 1, max_z + add_after)

    return min_z, max_z



def process_patient_data(ori_path,dataset_split,save_pat):
    """Iterate through the output directory and process data based on patient ID."""
    duration_lession = {}
    for folder_name in os.listdir(ori_path):
        folder_path = os.path.join(ori_path, folder_name)

        # Ensure it is a directory and check if it contains a patient ID
        if os.path.isdir(folder_path):
            assigned_split = None

            for split, patient_list in dataset_split.items():
                for patient_id in patient_list:
                    if str(patient_id) in folder_name:  # If patient_id appears in folder_name, match it
                        assigned_split = split
                        break
                if assigned_split:
                    break  # Once assigned, exit the loop

            if assigned_split is None:
                continue  # Skip if the patient does not belong to train/val/test

            # Define target folders
            image_target_dir = os.path.join(save_pat, assigned_split, "image")
            label_target_dir = os.path.join(save_pat, assigned_split, "label")

            # Locate the MR_t1.nii.gz image file
            mr_t1_path = find_mr_t1_file(folder_path)
            if not mr_t1_path:
                print(f"⚠️ Skipping {folder_name}: MR_t1.nii.gz not found")
                continue

            # Load MR image
            mr_t1_nii = nib.load(mr_t1_path)
            mr_t1_data = mr_t1_nii.get_fdata()
            total_slices = mr_t1_data.shape[2]  # Get total number of slices in z-axis

            # Process lesion labels
            lesion_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.endswith(".nii.gz") and not f.endswith("_dose.nii.gz") and not f.endswith("_MR_t1.nii.gz")
            ]

            if not lesion_files:
                print(f"⚠️ {folder_name} has no lesion, skipping")
                continue  # Skip if there are no lesion files

            for lesion_file in lesion_files:
                lesion_nii = nib.load(lesion_file)
                lesion_data = lesion_nii.get_fdata()

                # Compute the z-axis range of the lesion
                z_indices = np.any(lesion_data, axis=(0, 1))  # Identify z slices that contain the lesion
                min_z, max_z = np.where(z_indices)[0][[0, -1]]  # Get the first and last z slice index

                # **Expand the z-axis range to at least 16 slices**
                min_z, max_z = adjust_z_range(min_z, max_z, total_slices)

                # Extract expanded image slices
                cropped_img = mr_t1_data[:, :, min_z:max_z+1]
                cropped_label = lesion_data[:, :, min_z:max_z+1]

                # Generate new image file name
                lesion_base_name = os.path.basename(lesion_file).replace(".nii.gz", "")
                new_image_name = f"{lesion_base_name}_img.nii.gz"

                # Save the cropped image
                cropped_img_nii = nib.Nifti1Image(cropped_img, affine=mr_t1_nii.affine)
                nib.save(cropped_img_nii, os.path.join(image_target_dir, new_image_name))

                # Save the lesion label (keeping the z-axis alignment)
                cropped_lesion_nii = nib.Nifti1Image(cropped_label, affine=lesion_nii.affine)
                nib.save(cropped_lesion_nii, os.path.join(label_target_dir, f"{lesion_base_name}.nii.gz"))

                print(f"✅ Processed: {new_image_name}, corresponding label: {lesion_base_name}.nii.gz")


def process_patient_data_single(ori_path,dataset_split,save_pat):
    """Iterate through the output directory and process data based on patient ID."""
    duration_lession = {}
    for folder_name in os.listdir(ori_path):
        folder_path = os.path.join(ori_path, folder_name)

        # Ensure it is a directory and check if it contains a patient ID
        if os.path.isdir(folder_path):
            assigned_split = None

            for split, patient_list in dataset_split.items():
                for patient_id in patient_list:
                    if str(patient_id) in folder_name:  # If patient_id appears in folder_name, match it
                        assigned_split = split
                        break
                if assigned_split:
                    break  # Once assigned, exit the loop

            if assigned_split is None:
                continue  # Skip if the patient does not belong to train/val/test

            # Define target folders
            image_target_dir = os.path.join(save_pat, assigned_split, "image")
            label_target_dir = os.path.join(save_pat, assigned_split, "label")

            # Locate the MR_t1.nii.gz image file
            mr_t1_path = find_mr_t1_file(folder_path)
            if not mr_t1_path:
                print(f"⚠️ Skipping {folder_name}: MR_t1.nii.gz not found")
                continue

            # Load MR image
            mr_t1_nii = nib.load(mr_t1_path)
            mr_t1_data = mr_t1_nii.get_fdata()
            total_slices = mr_t1_data.shape[2]  # Get total number of slices in z-axis

            # Process lesion labels
            lesion_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.endswith(".nii.gz") and not f.endswith("_dose.nii.gz") and not f.endswith("_MR_t1.nii.gz")
            ]

            if not lesion_files:
                print(f"⚠️ {folder_name} has no lesion, skipping")
                continue  # Skip if there are no lesion files

            for lesion_file in lesion_files:
                lesion_nii = nib.load(lesion_file)
                lesion_data = lesion_nii.get_fdata()

                # # Compute the z-axis range of the lesion
                # z_indices = np.any(lesion_data, axis=(0, 1))  # Identify z slices that contain the lesion
                # min_z, max_z = np.where(z_indices)[0][[0, -1]]  # Get the first and last z slice index

                # # **Expand the z-axis range to at least 16 slices**
                # min_z, max_z = adjust_z_range(min_z, max_z, total_slices)

                # # Extract expanded image slices
                # cropped_img = mr_t1_data[:, :, min_z:max_z+1]
                # cropped_label = lesion_data[:, :, min_z:max_z+1]
                # change to no crop
                cropped_img = mr_t1_data
                cropped_label = lesion_data
                # Generate new image file name
                lesion_base_name = os.path.basename(lesion_file).replace(".nii.gz", "")
                new_image_name = f"{lesion_base_name}_img.nii.gz"

                # Save the cropped image
                cropped_img_nii = nib.Nifti1Image(cropped_img, affine=mr_t1_nii.affine)
                nib.save(cropped_img_nii, os.path.join(image_target_dir, new_image_name))

                # Save the lesion label (keeping the z-axis alignment)
                cropped_lesion_nii = nib.Nifti1Image(cropped_label, affine=lesion_nii.affine)
                nib.save(cropped_lesion_nii, os.path.join(label_target_dir, f"{lesion_base_name}.nii.gz"))

                print(f"✅ Processed: {new_image_name}, corresponding label: {lesion_base_name}.nii.gz")

if __name__=="__main__":
    # Define the file path
    file_path = "/Volumes/macdata/dataset/PKG - Brain-TR-GammaKnife/Brain-TR-GammaKnife-Clinical-Information.xlsx"
    dataset_split = split_data(file_path)

    ori_path= "/Volumes/macdata/dataset_read/brain_mri/brain_mri/output"
    save_path = "/Volumes/macdata/dataset_read/brain_mri/brain_mri/select_single"
    create_split_folders(save_path)
    # combine all label
    # process_dataset(ori_path,dataset_split,save_path)
    # crop
    process_patient_data(ori_path,dataset_split,save_path)
    # without crop
    process_patient_data_single(ori_path,dataset_split,save_path)



