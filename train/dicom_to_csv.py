import os
import sys
import pydicom
from tqdm import tqdm
import pandas as pd

ROOT_DIR = os.getcwd()
TRAIN_DIR = os.path.join(ROOT_DIR, 'stage_2_train_images')
TEST_DIR = os.path.join(ROOT_DIR, 'stage_2_test_images')
LABELS_DIR = os.path.join(ROOT_DIR, 'stage_2_train_labels.csv')

if __name__ == '__main__':

    PatientId = []
    Gender = []
    Age = []
    Class = []
    
    labels = pd.read_csv(LABELS_DIR)

    for index, row in tqdm(labels.iterrows()):
        patientId = row['patientId']
        target = row['Target']
        dcm = pydicom.read_file(os.path.join(TRAIN_DIR, patientId + '.dcm'))
        gender = dcm.PatientSex
        age = dcm.PatientAge

        PatientId.append(patientId)
        Gender.append(gender)
        Age.append(age)
        Class.append(target)

    df = pd.DataFrame({
        'PatientId': pd.Series(PatientId),
        'Age': pd.Series(Age),
        'Gender': pd.Series(Gender),
        'Class': pd.Series(Class)
    })

    df.to_csv(os.path.join(ROOT_DIR, 'PneumoniaDataset.csv'))