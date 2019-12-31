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
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from sklearn.model_selection import KFold
from pneumonia_model import InferenceConfig
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
    print(dicom_dir)
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations 

def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors

def Detect(model_path, dataset_dir, num_example):
    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(mode='inference', config=inference_config, model_dir=ROOT_DIR)
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    image_fps, image_annotations = parse_dataset(dataset_dir, pd.read_csv(train_labels_dir))
    image_fps_list = list(image_fps)
    random.shuffle(image_fps_list)
    image_fps_val = image_fps_list[:num_example]

    dataset = PneumoniaDataset(image_fps_val, image_annotations, 1024, 1024)
    dataset.prepare()
    fig = plt.figure(figsize=(10, 30))
    print("*****************")
    print(dataset.class_names)
    print("*****************")

    for i in range(num_example):

        image_id = random.choice(dataset.image_ids)
        
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, inference_config, 
                                image_id, use_mini_mask=False)
        
        print(original_image.shape)
        plt.subplot(6, 2, 2*i + 1)
        a = visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                    dataset.class_names, figsize=(32,32),
                                    colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
        plt.subplot(6, 2, 2*i + 2)
        results = model.detect([original_image]) #, verbose=1)
        print(results[0])
        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], figsize=(32, 32),
                                    colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
        
    plt.show()


# def Test(model_dir, dataset_dir, num_example):
#     print("#################################")
#     print(dataset_dir)
#     inference_config = InferenceConfig()
#     model = modellib.MaskRCNN(mode='inference', config=inference_config, model_dir=ROOT_DIR)
#     print("Loading weights from ", model_dir)
#     model.load_weights(model_dir, by_name=True)

#     dcmList = os.listdir(dataset_dir)
#     random.shuffle(dcmList)
#     dcmList = dcmList[:num_example]
#     for d in dcmList:
#         image = pydicom.read_file(os.path.join(dataset_dir, d)).pixel_array
#         image = cv2.resize(image, (256,256))
#         print(image.shape)
#         results = model.detect([image])
#         r = results[0]
#         img = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
#         cv2.imshow('image', img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

if __name__ == '__main__':
    train_dir = os.path.join(ROOT_DIR, sys.argv[1])
    epoch = sys.argv[2]
    dataset_dir = os.path.join(ROOT_DIR, sys.argv[3])
    print(dataset_dir)
    print("---")
    num_example = int(sys.argv[4])

    for d in os.listdir(train_dir):
        if epoch in d.replace('.h5',''):
            model_dir = os.path.join(train_dir, d)
    try:
        model_dir
    except NameError:
        print("Model not found")
    
    # Test(model_dir, dataset_dir, num_example)
    Detect(model_dir, dataset_dir, num_example)
    
