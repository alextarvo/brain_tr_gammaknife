import nrrd

import argparse
from types import SimpleNamespace
import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk
from radiomics import featureextractor


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
CLINICAL_INFO_LESION_COLUMNS = [PT_ID, LESION_COURSE_NO, LESION_NO, DURATION_TO_IMAG, TREATMENT_FRACTIONS, MRI_TYPE, LESION_FILE_NAME]
# This yields a list of string, that contains the _names_ of variables in CLINICAL_INFO_LESION_COLUMNS
CLINICAL_INFO_LESION_COLUMNS_NAMES = [name for name, value in globals().items() if value in CLINICAL_INFO_LESION_COLUMNS]
CLINICAL_INFO_COURSE_COLUMNS = [PT_ID, PATIENT_COURSE_NO, PATIENT_DIAGNOSIS_PRIMARY, PATIENT_AGE, PATIENT_GENDER]
CLINICAL_INFO_COURSE_COLUMNS_NAMES = [name for name, value in globals().items() if value in CLINICAL_INFO_COURSE_COLUMNS]

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


if __name__ == "__main__":
  args = get_args()
  df_merged = load_clinical_metadata(args.metadata_file)
  for index, row in df_merged.iterrows():
      patient_id = row['PT_ID']
      course_id = row['LESION_COURSE_NO']
      lesion_file_name = row['LESION_FILE_NAME']
      patient_course_id = f'GK.{patient_id}_{course_id}'
      print(f'Examining patient {patient_id}, course id {course_id}')

      # Read MRI image and its metadata. Note: here we use ITK as it seems to be a more
      # capable library.
      mri_path = os.path.join(args.nrrd_dataset_path, patient_course_id, f'{patient_course_id}_MR_t1.nrrd')
      mri_data, mri_image, mri_metadata = read_nrrd_and_metadata(mri_path)
      # We could use nrrd, as below. But all the examples are for ITK... so let''s go with ITK
      # tumor_data, tumor_header = nrrd.read(tumor_mask_path)

      # Read the tumor mask
      tumor_mask_path = os.path.join(args.nrrd_dataset_path, patient_course_id, f'{lesion_file_name}.nrrd')
      tumor_data, tumor_image, tumor_metadata = read_nrrd_and_metadata(mri_path)

      # Do some sanity check for images and metadata.
      np.testing.assert_array_equal(mri_image.shape, tumor_image.shape)
      assert(vars(mri_metadata) == vars(tumor_metadata))

      # Extract radiomics features. Do them all
      extractor = featureextractor.RadiomicsFeatureExtractor()
      features = extractor.execute(mri_path, tumor_mask_path)
      for key, value in features.items():
          print(f"{key}: {value}")
      print('done!')

