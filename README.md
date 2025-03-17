# Dataset Format and Training Pipeline (3D segmentation)

## 1. Prepare the Dataset

To convert the dataset into a format compatible with nnUNet, run the following command:

```bash
python segmentation_3d/nnunetv2/dataset_conversion/Dataset600_GK.py
```

## 2. Extract Dataset Fingerprint

A dataset fingerprint consists of dataset-specific properties such as image sizes, voxel spacings, intensity information, etc. This information is used to design three U-Net configurations. Run the following command to extract the fingerprint and verify dataset integrity:

```bash
nnUNetv2_plan_and_preprocess -d 600 --verify_dataset_integrity
```

## 3. Model Training

Train the model using the following commands. The model is trained on different GPUs as specified:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 3d_fullres 0 --npz & # Train on GPU 0
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 3d_fullres 1 --npz & # Train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 3d_fullres 2 --npz & # Train on GPU 1
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 3d_fullres 3 --npz & # Train on GPU 1
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 600 3d_fullres 4 --npz  # Train on GPU 1
```

## 4. Model Inference

To perform model inference, use the following command:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
