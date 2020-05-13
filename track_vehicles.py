import os, time, logging

import numpy as np
import cv2
import tensorflow as tf

# TF object detection imports
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# DeepSORT imports
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gen_det

# Custom imports
import custom_utils
from config import *


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def detect_vehicles(detection_model, frame):
    '''
    Returns:
        detection_list - list of bboxes in tlwh format
        detction_scores
    '''
    output_dict =  custom_utils.run_inference_for_single_image(detection_model, frame)

    all_bboxes = output_dict['detection_boxes']
    all_detection_classes = output_dict['detection_classes']
    all_detection_scores = output_dict['detection_scores']

    detection_list = [] # List of DeepSORT's Detection objects
    detection_scores = [] # List of score for all humans detected
    for i, box in enumerate(all_bboxes):
        det_class = all_detection_classes[i]
        det_score = all_detection_scores[i]
        if (det_class == 3 or det_class == 6 or det_class == 8) and det_score > MIN_CONFIDENCE: # If the detected object is a vehicle with a confidence of at least MIN_CONFIDENCE
            y1,x1,y2,x2 = int(box[0]*frame.shape[0]), int(box[1]*frame.shape[1]), int(box[2]*frame.shape[0]), int(box[3]*frame.shape[1])
            detection_list.append(tuple([x1,y1,x2-x1,y2-y1])) # tlwh format bboxes
            detection_scores.append(det_score)

    return detection_list, detection_scores



###############################################################################

if __name__ == '__main__':
    detection_model, category_index = custom_utils.load_detection_model(DETECTION_MODEL_NAME)
    logger.debug(category_index[3]) # Car
    logger.debug(category_index[6]) # Bus
    logger.debug(category_index[8]) # Truck

    frame = cv2.imread("./vehicle_img.png")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detection_list, detection_scores = detect_vehicles(detection_model, frame)

    for bbox in detection_list:
        x,y,w,h = bbox
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow('detected vehicles', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()