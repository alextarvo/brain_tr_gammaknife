import nrrd

import argparse
from types import SimpleNamespace
import os
import logging
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple
import random

import SimpleITK as sitk

import radiomics_extractor as ri

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logging.getLogger("radiomics").setLevel(logging.ERROR)

# column definitions in the clinical info Excel file
# This is for the lesion sheet
PT_ID = 'unique_pt_id'
LESION_COURSE_NO = 'Treatment Course'
LESION_NO = 'Lesion #'
DURATION_TO_IMAG = 'duration_tx_to_imag (months)'
TREATMENT_FRACTIONS = 'Fractions'
MRI_TYPE = 'mri_type'
LESION_FILE_NAME = 'Lesion Name in NRRD files'
# This is for the course sheet
PATIENT_COURSE_NO = 'Course #'
PATIENT_DIAGNOSIS_METS = 'Diagnosis (Only want Mets)'
PATIENT_DIAGNOSIS_PRIMARY = 'Primary Diagnosis'
PATIENT_AGE = 'Age at Diagnosis'
PATIENT_GENDER = 'Gender'

# Set of column names to read from the lesion file
CLINICAL_INFO_LESION_COLUMNS = [PT_ID, LESION_COURSE_NO, LESION_NO, DURATION_TO_IMAG, TREATMENT_FRACTIONS, MRI_TYPE,
                                LESION_FILE_NAME]
# This yields a list of string, that contains the _names_ of variables in CLINICAL_INFO_LESION_COLUMNS
CLINICAL_INFO_LESION_COLUMNS_NAMES = [name for name, value in globals().items() if
                                      value in CLINICAL_INFO_LESION_COLUMNS]
CLINICAL_INFO_COURSE_COLUMNS = [PT_ID, PATIENT_COURSE_NO, PATIENT_DIAGNOSIS_METS, PATIENT_DIAGNOSIS_PRIMARY,
                                PATIENT_AGE, PATIENT_GENDER]
CLINICAL_INFO_COURSE_COLUMNS_NAMES = [name for name, value in globals().items() if
                                      value in CLINICAL_INFO_COURSE_COLUMNS]

MIN_PATCH_DIM = 20


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)

@dataclass
class LesionInfo:
    patient_id: int
    lesion_no: int
    lesion_course_no: int
    mri_image: np.ndarray
    lesion_dimensions: np.ndarray
    mri_lesion_image: np.ndarray
    tumor_mask_image: np.ndarray


def get_args():
    """
       Sets up command line arguments parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_file",
                        help='Path to the clinical info file Brain-TR-GammaKnife-Clinical-Information.xlsx',
                        type=str)
    parser.add_argument("--nrrd_dataset_path",
                        help='Path to the dataset with nrrd files',
                        type=str)
    parser.add_argument("--metrics_output_path",
                        help='Path to the write the .csv file with radiomics features',
                        type=str)
    parser.add_argument("--patches_output_path",
                        help='Path to the write the .npy files that contain extracted patches with tumors',
                        type=str)
    parser.add_argument("--radiomics_extractor", help='Type of a radiomics metrics extractor to use',
                        type=str, choices=['pyradiomics', 'fcib', 'fcib_uptuned', 'none'], default='pyradiomics')
    parser.add_argument("--output_patches_path", help='The path where we may output patches',
                        type=str, default='')
    parser.add_argument("--fcib_uptuned_weights", help='The path where the weights of FCIB uptuned model are saved',
                        type=str, default='')
    args = parser.parse_args()
    return args


def load_clinical_metadata(metadata_file):
    """ Reads .xlsx file with the clinical summary. Joins data
    on a lesion level with treatment-course level data. Returns a dataframe
    whose columns are the union of both

    NOTE: original column names are the mess. Thus we use CLINICAL_INFO_LESION_COLUMNS_NAMES
    and CLINICAL_INFO_COURSE_COLUMNS_NAMES as names for a united dataframe.
    """
    # Read sheets from excel file
    df_clinical_info_lesions = pd.read_excel(metadata_file, sheet_name='lesion_level')[CLINICAL_INFO_LESION_COLUMNS]
    df_clinical_info_course = pd.read_excel(args.metadata_file, sheet_name='course_level')[CLINICAL_INFO_COURSE_COLUMNS]
    column_mapping_lesions = dict(zip(CLINICAL_INFO_LESION_COLUMNS, CLINICAL_INFO_LESION_COLUMNS_NAMES))
    column_mapping_course = dict(zip(CLINICAL_INFO_COURSE_COLUMNS, CLINICAL_INFO_COURSE_COLUMNS_NAMES))
    df_clinical_info_lesions.rename(columns=column_mapping_lesions, inplace=True)
    df_clinical_info_course.rename(columns=column_mapping_course, inplace=True)

    df_merged = df_clinical_info_lesions.merge(
        df_clinical_info_course,
        left_on=['PT_ID', 'LESION_COURSE_NO'],
        right_on=['PT_ID', 'PATIENT_COURSE_NO'])
    df_merged.drop(columns=['PATIENT_COURSE_NO'], inplace=True)
    return df_merged


def read_nrrd_and_metadata(filepath):
    """Read NRRD file and its metadata using ITK"""
    data = sitk.ReadImage(filepath)
    image = sitk.GetArrayFromImage(data)
    metadata_dict = {
        'spacing': data.GetSpacing(),
        'origin': data.GetOrigin(),
        'direction': data.GetDirection(),
    }
    # Convert it to the plain Python object for easier handling and better code.
    metadata = SimpleNamespace(**metadata_dict)
    return data, image, metadata


def extract_central_slices(mri_image, tumor_mask):
    def correct_dimensions(min_i, max_i, img_boundary_i):
        patch_dim = max_i - min_i
        center_slice = (min_i + max_i) // 2
        if patch_dim < MIN_PATCH_DIM:
            min_i = center_slice - MIN_PATCH_DIM // 2
            max_i = center_slice + MIN_PATCH_DIM // 2
        if min_i < 0:
            min_i = 0
            max_i = MIN_PATCH_DIM
        if min_i >= img_boundary_i:
            max_i = img_boundary_i - 1
            min_i = max_i - MIN_PATCH_DIM
        return center_slice, min_i, max_i

    # Alternative option - may be more compact, but less readable and may not be correct
    # nonzero_coords = np.argwhere(tumor_mask > 0)
    # # Get min/max indices for each axis
    # minD, minH, minW = nonzero_coords.min(axis=0)
    # maxD, maxH, maxW = nonzero_coords.max(axis=0)

    mask_shape = tumor_mask.shape
    min_1 = None
    max_1 = -1000
    for i in range(mask_shape[0]):
        if np.count_nonzero(tumor_mask[i, :, :]) > 0 and min_1 is None:
            min_1 = i
        if np.count_nonzero(tumor_mask[i, :, :]) > 0 and max_1 < i:
            max_1 = i + 1
    center_1, minc_1, maxc_1 = correct_dimensions(min_1, max_1, mask_shape[0])

    min_2 = None
    max_2 = -1000
    for i in range(mask_shape[1]):
        if np.count_nonzero(tumor_mask[:, i, :]) > 0 and min_2 is None:
            min_2 = i
        if np.count_nonzero(tumor_mask[:, i, :]) > 0 and max_2 < i:
            max_2 = i + 1
    center_2, minc_2, maxc_2 = correct_dimensions(min_2, max_2, mask_shape[1])

    min_3 = None
    max_3 = -1000
    for i in range(mask_shape[2]):
        if np.count_nonzero(tumor_mask[:, :, i]) > 0 and min_3 is None:
            min_3 = i
        if np.count_nonzero(tumor_mask[:, :, i]) > 0 and max_3 < i:
            max_3 = i + 1
    center_3, minc_3, maxc_3 = correct_dimensions(min_3, max_3, mask_shape[2])

    # show_slice(mri_image[center_1, :, :], None)
    # show_slice(mri_image[center_1, :, :], tumor_mask[center_1, :, :])
    # show_slice(mri_image[:, center_2, :], None)
    # show_slice(mri_image[:, center_2, :], tumor_mask[:, center_2, :])
    # show_slice(mri_image[:, :, center_3], None)
    # show_slice(mri_image[:, :, center_3], tumor_mask[:, :, center_3])

    center = (center_1, center_2, center_3)
    # dims_corrected[0] min/max  coordinates along a first dimension
    dims_corrected = np.array(((minc_1, maxc_1), (minc_2, maxc_2), (minc_3, maxc_3)))
    dims_uncorrected = np.array(((min_1, max_1), (min_2, max_2), (min_3, max_3)))

    return center, dims_uncorrected, dims_corrected


def show_slice(slice, mask=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(slice, cmap="gray")
    if mask is not None:
        plt.imshow(mask, cmap="Blues", alpha=0.2)
    plt.axis("off")
    plt.show()


def compare_metadata(metadata1, metadata2):
    return metadata1 == metadata2


def boxes_intersect(box1: np.array, box2: np.array) -> bool:
    """
    Checks if two 3D bounding boxes intersect.

    Args:
        box1: First box ((x_min, x_max), (y_min, y_max), (z_min, z_max)) or a NumPy array.
        box2: Second box ((x_min, x_max), (y_min, y_max), (z_min, z_max)) or a NumPy array.

    Returns:
        bool: True if the boxes intersect, False otherwise.
    """
    # Check for overlap in all three dimensions
    x_overlap = box1[0, 0] < box2[0, 1] and box1[0, 1] > box2[0, 0]
    y_overlap = box1[1, 0] < box2[1, 1] and box1[1, 1] > box2[1, 0]
    z_overlap = box1[2, 0] < box2[2, 1] and box1[2, 1] > box2[2, 0]
    return x_overlap and y_overlap and z_overlap


def create_finetuning_dataset(dataset_path, lesion_infos, lesion_free_factor, prob_train=0.75):
    def show_lesion(lesion_image, tumor_mask_image, lesion_dimensions):
        mid_section = ((lesion_dimensions[:, 1]-lesion_dimensions[:,0])/2).astype(int)
        if tumor_mask_image is not None:
            show_slice(lesion_image[mid_section[0], :, :], tumor_mask_image[mid_section[0],:,:])
            show_slice(lesion_image[:, mid_section[1], :], tumor_mask_image[:, mid_section[1], :])
            show_slice(lesion_image[:, :, mid_section[2]], tumor_mask_image[:, :, mid_section[2]])
        else:
            show_slice(lesion_image[mid_section[0], :, :], None)
            show_slice(lesion_image[:, mid_section[1], :], None)
            show_slice(lesion_image[:, :, mid_section[2]], None)

    if dataset_path is not None:
        os.makedirs(dataset_path, exist_ok=True)
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for lesion_idx in range(len(lesion_infos)):
        test_flag = random.uniform(0, 1)
        path = train_path if test_flag < prob_train else test_path

        lesion = lesion_infos[lesion_idx]
        lesion_id = f'{lesion.patient_id}_{lesion.lesion_no}_{lesion.lesion_course_no}'
        # print(lesion_id)
        lesion_dims = lesion.lesion_dimensions

        # Debug only
        # show_lesion(lesion.mri_lesion_image, lesion.tumor_mask_image, lesion_dims)

        pat_id_used = {lesion.patient_id}
        np.save(os.path.join(path, f'lesion_{lesion_id}.npy'), lesion.mri_lesion_image)
        num_lesion_free_generated = 0
        while num_lesion_free_generated < lesion_free_factor:
            other_idx = random.randint(0, len(lesion_infos) - 1)
            other_lesion = lesion_infos[other_idx]
            if other_lesion.patient_id in pat_id_used:
                # Do NOT  use the same patient for getting a  "lesion-free" image - as lesion instances are per tumor.
                # It is possible that we will pick the MRI image for same person/treatment, just a different lesion id.
                # In this case, we will "cut" the image of the "current" tumor - just from other_lesion instance.
                continue
            if boxes_intersect(lesion.lesion_dimensions, other_lesion.lesion_dimensions):
                # There's a lesion in another image in exactly same area for a different patient!
                continue
            other_lesion_id = f'{other_lesion.patient_id}_{other_lesion.lesion_no}_{other_lesion.lesion_course_no}'
            lesion_free_slice = other_lesion.mri_image[
                                lesion_dims[0][0]:lesion_dims[0][1],
                                lesion_dims[1][0]:lesion_dims[1][1],
                                lesion_dims[2][0]: lesion_dims[2][1]]
            # Debug only
            # show_lesion(lesion_free_slice, None, lesion_dims)
            lesion_free_id = f'{lesion_id}_{other_lesion_id}__{num_lesion_free_generated}'
            np.save(os.path.join(path, f'free_{lesion_free_id}.npy'), lesion_free_slice)
            num_lesion_free_generated += 1
            pat_id_used.add(other_lesion.patient_id)


if __name__ == "__main__":
    args = get_args()
    df_merged = load_clinical_metadata(args.metadata_file)
    if args.radiomics_extractor == 'pyradiomics':
        extractor = ri.PyRadiomicsExtractor()
    elif args.radiomics_extractor == 'fcib':
        extractor = ri.FCIBImageExtractor()
    elif args.radiomics_extractor == 'fcib_uptuned':
        extractor = ri.FCIBTunedImageExtractor(args.fcib_uptuned_weights)
    else:
        extractor = None

    seed_everything()

    metrics = []

    lesions_infos = []

    prev_mri_metadata = None
    prev_tumor_metadata = None

    if args.patches_output_path is not None:
        os.makedirs(os.path.dirname(args.patches_output_path), exist_ok=True)

    for index, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Processing lesions"):
        patient_id = row['PT_ID']
        lesion_id = row['LESION_NO']
        course_id = row['LESION_COURSE_NO']
        lesion_file_name = row['LESION_FILE_NAME']
        patient_course_id = f'GK.{patient_id}_{course_id}'

        logging.info(f'Examining patient {patient_id}, course id {course_id}')
        # Debug only
        # if row['MRI_TYPE'] != 'recurrence':
        #     continue

        # Read MRI image and its metadata. Note: here we use ITK as it seems to be a more
        # capable library.
        mri_path = os.path.join(args.nrrd_dataset_path, patient_course_id, f'{patient_course_id}_MR_t1.nrrd')
        tumor_mask_path = os.path.join(args.nrrd_dataset_path, patient_course_id, f'{lesion_file_name}.nrrd')
        # print(f'Loading MRI file {mri_path}, tumor mask: {tumor_mask_path}')

        if not os.path.exists(mri_path):
            logging.warning(f'MRI file {mri_path} does not exist!')
        if not os.path.exists(tumor_mask_path):
            logging.warning(f'Tumor file {tumor_mask_path} does not exist!')

        # We could use nrrd, as below. But all the examples are for ITK... so let''s go with ITK
        # tumor_data, tumor_header = nrrd.read(tumor_mask_path)

        # Read NRRD.
        mri_data, mri_image, mri_metadata = read_nrrd_and_metadata(mri_path)
        # Read the tumor mask
        tumor_data, tumor_image, tumor_metadata = read_nrrd_and_metadata(tumor_mask_path)
        # Do some sanity check for images and metadata.
        np.testing.assert_array_equal(mri_image.shape, tumor_image.shape)
        assert (vars(mri_metadata) == vars(tumor_metadata))

        # Make sure metadata same across images
        if prev_mri_metadata is not None:
            if not compare_metadata(mri_metadata, prev_mri_metadata):
                logging.warning(f'MRI metadata changed from {prev_mri_metadata} to {mri_metadata}')
        prev_mri_metadata = mri_metadata

        if prev_tumor_metadata is not None:
            if not compare_metadata(mri_metadata, prev_mri_metadata):
                logging.warning(f'Mask metadata changed from {prev_tumor_metadata} to {tumor_metadata}')
        prev_tumor_metadata = tumor_metadata

        if not compare_metadata(mri_metadata, tumor_metadata):
            logging.warning(f'Mask metadata changed from {mri_metadata} to {tumor_metadata}')

        # Get image slices that contain data
        center, dims_uncorrected, dims_corrected = extract_central_slices(mri_image, tumor_image)

        extracted_mri_slice = mri_image[
                              dims_corrected[0][0]:dims_corrected[0][1],
                              dims_corrected[1][0]:dims_corrected[1][1],
                              dims_corrected[2][0]: dims_corrected[2][1]
                              ]
        extracted_tumor_slice = tumor_image[
                              dims_corrected[0][0]:dims_corrected[0][1],
                              dims_corrected[1][0]:dims_corrected[1][1],
                              dims_corrected[2][0]: dims_corrected[2][1]
                              ]

        lesion = LesionInfo(patient_id, lesion_id, course_id,
                            mri_image, dims_corrected, extracted_mri_slice, extracted_tumor_slice)
        lesions_infos.append(lesion)
        # entry_patch = {
        #     'PT_ID': row['PT_ID'],
        #     'LESION_COURSE_NO': row['LESION_COURSE_NO'],
        #     'LESION_NO': row['LESION_NO'],
        #     'SLICE_FILE': patient_lesion_course_file
        # }

        # Extract radiomics features. Do them all
        try:
            PT_ID = 'unique_pt_id'
            LESION_COURSE_NO = 'Treatment Course'
            LESION_NO = 'Lesion #'
            DURATION_TO_IMAG = 'duration_tx_to_imag (months)'
            TREATMENT_FRACTIONS = 'Fractions'
            MRI_TYPE = 'mri_type'
            LESION_FILE_NAME = 'Lesion Name in NRRD files'
            # This is for the course sheet
            PATIENT_COURSE_NO = 'Course #'
            PATIENT_DIAGNOSIS_METS = 'Diagnosis (Only want Mets)'
            PATIENT_DIAGNOSIS_PRIMARY = 'Primary Diagnosis'
            PATIENT_AGE = 'Age at Diagnosis'
            PATIENT_GENDER = 'Gender'

            entry = {
                # 'LESION_KEY' = f'{row["PT_ID"]}_{row["LESION_COURSE_NO"]}_{row["LESION_NO"]}',
                'PT_ID': row['PT_ID'],
                'LESION_COURSE_NO': row['LESION_COURSE_NO'],
                'LESION_NO': row['LESION_NO'],
                'PATIENT_DIAGNOSIS_METS': row['PATIENT_DIAGNOSIS_METS'],
                'PATIENT_DIAGNOSIS_PRIMARY': row['PATIENT_DIAGNOSIS_PRIMARY'],
                'PATIENT_AGE': row['PATIENT_AGE'],
                'PATIENT_GENDER': row['PATIENT_GENDER'],
                'DURATION_TO_IMAG': row['DURATION_TO_IMAG'],
                'MRI_TYPE': row['MRI_TYPE'],
            }
            if extractor is not None:
                filtered_metrics = extractor.extract_features(
                    mri_path=mri_path, tumor_mask_path=tumor_mask_path,
                    image_patch=extracted_mri_slice)
                entry.update(filtered_metrics)
                metrics.append(entry)

        except:
            logging.exception(
                f"Error while extracting features from the MRI file {mri_path}, tumor mask file {tumor_mask_path}")

    if len(metrics) > 0:
        df_metrics = pd.DataFrame(metrics)
        if args.metrics_output_path is not None:
            os.makedirs(os.path.dirname(args.metrics_output_path), exist_ok=True)
        df_metrics.to_csv(args.metrics_output_path)

    if args.output_patches_path is not None and args.output_patches_path != '':
        print(f'Outputting finetuning dataset to {args.output_patches_path}')
        create_finetuning_dataset(args.output_patches_path, lesions_infos, 3)

    logging.info('Done processing!')
