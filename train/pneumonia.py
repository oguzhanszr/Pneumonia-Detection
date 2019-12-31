"""------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 pneumonia.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 pneumonia.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 pneumonia.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 pneumonia.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.

import matplotlib.pyplot as plt

import os
import sys
import random
import math
import cv2
import pydicom
from tqdm import tqdm
import pandas as pd
import glob
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
from sklearn.model_selection import KFold
from pneumonia_model import PneumoniaConfig
from pneumonia_model import PneumoniaDataset
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
train_dicom_dir = os.path.join(ROOT_DIR, "samples\pneumonia\stage_2_train_images")
test_dicom_dir = os.path.join(ROOT_DIR, "samples\pneumonia\stage_2_test_images")
train_labels_dir = os.path.join(ROOT_DIR, "samples\pneumonia\stage_2_train_labels.csv")


def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))
def parse_dataset(dicom_dir, anns): 
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations 



config = PneumoniaConfig()
config.display()



anns = pd.read_csv(train_labels_dir)
print(anns.head())

image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)
ds = pydicom.read_file(image_fps[0]) # read dicom image from filepath 
image = ds.pixel_array # get image array

# Original image size: 1024 x 1024
ORIG_SIZE = 1024

image_fps_list = list(image_fps)
random.seed(42)
random.shuffle(image_fps_list)
val_size = len(image_fps_list) * 0.3
image_fps_val = image_fps_list[:val_size]
image_fps_train = image_fps_list[val_size:]

print("len(Train) - len(Val)")
print(len(image_fps_train), len(image_fps_val))

dataset_train = PneumoniaDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

dataset_val = PneumoniaDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()


model = modellib.MaskRCNN(mode='training', config = config, model_dir=ROOT_DIR)

#IMAGE NET 
model.load_weights(model.get_imagenet_weights(), by_name=True)

LEARNING_RATE = 0.005
import warnings 
warnings.filterwarnings("ignore")

#Train schedule
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE,
            epochs=20,
            layers='all',
            augmentation=None)
history = model.keras_model.history.history

best_epoch = np.argmin(history["val_loss"])
print("Best Epoch:", best_epoch + 1, history["val_loss"][best_epoch])

epochs = range(1,len(next(iter(history.values())))+1)
df = pd.DataFrame(history, index=epochs)
df.to_csv (ROOT_DIR+'history.csv', index = None, header=True)
