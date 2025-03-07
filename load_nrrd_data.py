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

import SimpleITK as sitk
from radiomics import featureextractor

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

    mask_shape = tumor_mask.shape
    min_1 = None
    max_1 = -1000
    for i in range(mask_shape[0]):
        if np.count_nonzero(tumor_mask[i, :, :]) > 0 and min_1 is None:
            min_1 = i
        if np.count_nonzero(tumor_mask[i, :, :]) > 0 and max_1 < i:
            max_1 = i
    center_1, min_1, max_1 = correct_dimensions(min_1, max_1, mask_shape[0])

    min_2 = None
    max_2 = -1000
    for i in range(mask_shape[1]):
        if np.count_nonzero(tumor_mask[:, i, :]) > 0 and min_2 is None:
            min_2 = i
        if np.count_nonzero(tumor_mask[:, i, :]) > 0 and max_2 < i:
            max_2 = i
    center_2, min_2, max_2 = correct_dimensions(min_2, max_2, mask_shape[1])

    min_3 = None
    max_3 = -1000
    for i in range(mask_shape[2]):
        if np.count_nonzero(tumor_mask[:, :, i]) > 0 and min_3 is None:
            min_3 = i
        if np.count_nonzero(tumor_mask[:, :, i]) > 0 and max_3 < i:
            max_3 = i
    center_3, min_3, max_3 = correct_dimensions(min_3, max_3, mask_shape[2])
    #
    # show_slice(mri_image[center_1, :, :], None)
    # show_slice(mri_image[center_1, :, :], tumor_mask[center_1, :, :])
    # show_slice(mri_image[:, center_2, :], None)
    # show_slice(mri_image[:, center_2, :], tumor_mask[:, center_2, :])
    # show_slice(mri_image[:, :, center_3], None)
    # show_slice(mri_image[:, :, center_3], tumor_mask[:, :, center_3])

    return mri_image[min_1:max_1, min_2:max_2, min_3:max_3]


def show_slice(slice, mask=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(slice, cmap="gray")
    if mask is not None:
        plt.imshow(mask, cmap="Blues", alpha=0.3)
    plt.axis("off")
    plt.show()


def compare_metadata(metadata1, metadata2):
    return metadata1 == metadata2


if __name__ == "__main__":
    args = get_args()
    df_merged = load_clinical_metadata(args.metadata_file)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    # df_output = pd.DataFrame()

    metrics = []
    extracted_patches = []

    prev_mri_metadata = None
    prev_tumor_metadata = None

    # for index, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Processing lesions"):
    for index, row in df_merged.iterrows():
        patient_id = row['PT_ID']
        course_id = row['LESION_COURSE_NO']
        lesion_file_name = row['LESION_FILE_NAME']
        patient_course_id = f'GK.{patient_id}_{course_id}'
        logging.info(f'Examining patient {patient_id}, course id {course_id}')

        # Read MRI image and its metadata. Note: here we use ITK as it seems to be a more
        # capable library.
        mri_path = os.path.join(args.nrrd_dataset_path, patient_course_id, f'{patient_course_id}_MR_t1.nrrd')
        tumor_mask_path = os.path.join(args.nrrd_dataset_path, patient_course_id, f'{lesion_file_name}.nrrd')
        print(f'Loading MRI file {mri_path}, tumor mask: {tumor_mask_path}')

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
        tumor_slices = extract_central_slices(mri_image, tumor_image)

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
            entry_patch = {

            }

            features = extractor.execute(mri_path, tumor_mask_path)
            # for key, value in features.items():
            #     print(f"{key}: {value}")
            filtered_metrics = {k.replace("original_", ""): v for k, v in features.items() if k.startswith("original_")}

            for key in filtered_metrics.keys():
                value = filtered_metrics[key]
                if (isinstance(value, np.ndarray) and value.shape == ()) or (
                        isinstance(value, list) and len(value) == 1):
                    value = float(value)
                    filtered_metrics[key] = value
            entry.update(filtered_metrics)
            metrics.append(entry)

        except:
            logging.exception(
                f"Error while extracting features from the MRI file {mri_path}, tumor mask file {tumor_mask_path}")

    df_metrics = pd.DataFrame(metrics)

    if args.metrics_output_path is not None:
        os.makedirs(os.path.dirname(args.metrics_output_path), exist_ok=True)
    df_metrics.to_csv(args.metrics_output_path)
    logging.info('Done processing!')
