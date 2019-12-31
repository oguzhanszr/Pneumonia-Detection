import os
import sys
import random
import math
import cv2
import base64
import pydicom
from tqdm import tqdm
import pandas as pd
import glob
import json
import datetime
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO
from io import BytesIO
from imgaug import augmenters as iaa
from sklearn.model_selection import KFold
from keras import backend as K
os.chdir('../')
ROOT_DIR = os.getcwd()
sys.path.append(os.path.join(str(os.getcwd()), 'model'))
from pneumonia_model import InferenceConfig
from pneumonia_model import PneumoniaDataset
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
os.chdir(os.path.join(str(os.getcwd()), 'api'))

MODEL_DIR = os.path.join(ROOT_DIR, 'model\\model.h5')

def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


def predict(img):
    encoded_img = str(img).split(',')[1]
    decoded_img = base64.b64decode(encoded_img)
    img = Image.open(BytesIO(decoded_img))
    img = np.asarray(img, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    K.clear_session()

    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode = 'inference', config = inference_config, model_dir = MODEL_DIR)
    print('Reading model from {}'.format(MODEL_DIR))
    model.load_weights(MODEL_DIR, by_name=True)

    # deneme =os.path.join(ROOT_DIR,"model\\test.jpg")
    # print("ISSOSOSOS : "+deneme)
    # print("ASODKSADKSA :" +os.listdir(deneme))
    img = cv2.resize(img, (256,256))
    results = model.detect([img])
    r = results[0]
    # print(r['class_ids'])
    mask = visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                    ['BG', 'pneumonia'], r['scores'], figsize=(32, 32),
                                    colors=get_colors_for_class_ids(r['class_ids']))
    if type(mask) != type(img):
        return -1, -1
    else:
        mask = Image.fromarray(mask.astype("uint8"))
        rawBytes = BytesIO()
        mask.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        # encoded_string = base64.b64encode(mask)
        confMean = 0
        count = 0

        for i in r['scores']:
            print(i)
            confMean = confMean + float(i)
            count = count + 1
        confidence = float(confMean / count * 100)
        return str(img_base64), str(confidence)


