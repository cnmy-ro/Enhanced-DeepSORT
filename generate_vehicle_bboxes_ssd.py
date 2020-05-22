import os, sys, logging

import numpy as np
import cv2
import tensorflow as tf

#sys.path.append('../')
from object_detection.utils import ops as utils_ops

import custom_utils
from config import *


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('gen_ssd_bboxes')

###############################################################################

def detect_vehicles(detection_model, frame):
    '''
    Returns:
        detection_list - list of bboxes in TLWH format
        detction_scores
    '''
    output_dict =  custom_utils.run_inference_for_single_image(detection_model, frame)

    all_bboxes = output_dict['detection_boxes']
    all_detection_classes = output_dict['detection_classes']
    all_detection_scores = output_dict['detection_scores']

    bbox_list = [] # List of TLWH bboxes
    detection_scores = [] # List of score for all humans detected
    for i, box in enumerate(all_bboxes):
        det_class = all_detection_classes[i]
        det_score = all_detection_scores[i]
        # If the detected object is a vehicle
        if (det_class == 3 or det_class == 6 or det_class == 8):
            y1,x1,y2,x2 = int(box[0]*frame.shape[0]), int(box[1]*frame.shape[1]), int(box[2]*frame.shape[0]), int(box[3]*frame.shape[1])
            bbox_list.append(tuple([x1,y1,x2-x1,y2-y1])) # TLWH format bboxes
            detection_scores.append(det_score)

    return bbox_list, detection_scores


def generate_bboxes(vehicle_data_dir, detection_model, output_dir):
    for sequence in sorted(os.listdir(vehicle_data_dir))[9:]: #####
        logger.debug("Processing %s" % sequence)
        sequence_dir = os.path.join(vehicle_data_dir, sequence)
        # image_dir = os.path.join(sequence_dir, "img1")
        # image_filenames = {int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        #                    for f in os.listdir(image_dir)}

        image_filenames  = {i+1 : os.path.join(sequence_dir,f) for i, f in enumerate(sorted(os.listdir(sequence_dir)))}

        output_file_path = output_dir + sequence + '.txt'

        output_file = open(output_file_path, 'w')
        for frame_idx in sorted(image_filenames.keys()):
            logger.debug("Processing frame {}".format(frame_idx))
            frame = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox_list, detection_scores = detect_vehicles(detection_model, frame_rgb)
            for i, bbox in enumerate(bbox_list):
                score = detection_scores[i]
                # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), 1)
                output_file.write("{},-1,{},{},{},{},{:.2f}\n".format(frame_idx, bbox[0], bbox[1], bbox[2], bbox[3], score))
            # cv2.imshow('bboxes', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        output_file.close()

###############################################################################

if __name__ == '__main__':
    vehicle_data_dir = "UA-DETRAC/Insight-MVT_Annotation_Train/"
    detection_model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    # detection_model_dir = 'object_detection/models/'+ model_name + '/saved_model'
    output_dir = "UA-DETRAC/Object Data/Bboxes/SSD/"

    detection_model, category_index = custom_utils.load_detection_model(detection_model_name)
    generate_bboxes(vehicle_data_dir, detection_model, output_dir)
