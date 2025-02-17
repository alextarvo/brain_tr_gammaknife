import pydicom

import numpy as np

import os
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def show_slice(slice):
    plt.imshow(slice, cmap="gray")
    plt.axis("off")
    plt.show()

# A path to the GammaKnife ddataset on your PC.
GAMMAKNIFE_DATA_PATH = '/mnt/data/GammaKnife/manifest-1678464337678/Brain-TR-GammaKnife/'

def load_dicom_mri_images(folder_path):
    dicom_files = sorted(glob(f'{folder_path}/*.dcm'), key=lambda f: pydicom.dcmread(f).InstanceNumber)
    mri_records = [pydicom.dcmread(f) for f in dicom_files]
    image_stack = np.stack([mri_record.pixel_array for mri_record in mri_records], axis=0)
    return mri_records, image_stack

# here, is just some small script to load DICOM images into the Numpy arrays from a given MRI session.
# Path to ffolder where the images are created
sample_mri_folder_path = os.path.join(GAMMAKNIFE_DATA_PATH, 'GK_103/04-18-2014-NA-MR GAMMA KNIFE PLANNING BRAIN W IV CONTRAST-49648/4.000000-t1mprtraiso-40897')

 # Read images and metadata
mri_records, image_stack = load_dicom_mri_images(sample_mri_folder_path)
# Print patient position in a coordinate frame of a MRI scan
for mri_record in mri_records:
    print(f'Coordinates of the {mri_record.InstanceNumber} slice of MRI: {mri_record.ImagePositionPatient}')

# Just show some slice of an MRI image
print(f'Resulting shape of an MRI image, as Numnpy array: {image_stack.shape}')
show_slice(image_stack[100, :, :])
show_slice(image_stack[:, 100, :])
show_slice(image_stack[:, :, 100])

# Load a DICOM file with radiotherapy planning
sample_rt_planning_file = os.path.join(GAMMAKNIFE_DATA_PATH, 'GK_103/04-18-2014-NA-MR GAMMA KNIFE PLANNING BRAIN W IV CONTRAST-49648/28641.000000-Comp Dose-71122')
dicom_record = pydicom.dcmread(f"{sample_rt_planning_file}/1-1.dcm")
# For dose files, we do have everything in a single 3D array
dose_stack = dicom_record.pixel_array

# Just show some slice of an RT dose mask
print(f'Resulting shape of a RT treatment plan, as Numnpy array: {dose_stack.shape}')
show_slice(dose_stack[100, :, :])
show_slice(dose_stack[:, 100, :])
show_slice(dose_stack[:, :, 100])

print(f'Coordinates of the 0th (??) slice of RT plan: {dicom_record.ImagePositionPatient}')