import os, sys, logging

import numpy as np
import cv2
import tensorflow as tf

sys.path.append('../')
from object_detection.utils import ops as utils_ops

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('gen_ssd_bboxes')

###############################################################################

def load_detection_model(model_dir):
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy()
                   for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
      # Reframe the the bbox mask to the image size.
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'],
                                                                            output_dict['detection_boxes'],
                                                                            image.shape[0],
                                                                            image.shape[1])
      detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                         tf.uint8)
      output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def detect_humans(detection_model, frame, min_confidence):
    output_dict =  run_inference_for_single_image(detection_model, frame)

    all_bboxes = output_dict['detection_boxes']
    all_detection_classes = output_dict['detection_classes']
    all_detection_scores = output_dict['detection_scores']

    detection_list = [] # List of DeepSORT's Detection objects
    detection_scores = [] # List of score for all humans detected
    for i, box in enumerate(all_bboxes):
        det_class = all_detection_classes[i]
        det_score = all_detection_scores[i]
        if det_class == 1 and det_score > min_confidence: # If the detected object is a person with a confidence of at least MIN_CONFIDENCE
            y1,x1,y2,x2 = int(box[0]*frame.shape[0]), int(box[1]*frame.shape[1]), int(box[2]*frame.shape[0]), int(box[3]*frame.shape[1])
            detection_list.append(tuple([x1,y1,x2-x1,y2-y1]))
            detection_scores.append(det_score)

    return detection_list, detection_scores


def generate_bboxes(mot_dir, detection_model, output_custom_det_dir):
    for sequence in os.listdir(mot_dir):
        logger.debug("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)
        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
                           for f in os.listdir(image_dir)}
        output_file_path = os.path.join(output_custom_det_dir, sequence, 'det/det.txt')
        output_file = open(output_file_path, 'w')
        for frame_idx in sorted(image_filenames.keys()):
            logger.debug("Processing frame {}".format(frame_idx))
            frame = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox_list, detection_scores = detect_humans(detection_model, frame_rgb, min_confidence=0.5)
            for bbox in bbox_list:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), 1)
                output_file.write("{},-1,{},{},{},{},1,-1,-1,-1\n".format(frame_idx, bbox[0], bbox[1], bbox[2], bbox[3]))
            # cv2.imshow('bboxes', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        output_file.close()

###############################################################################

if __name__ == '__main__':
    mot_dir = '../MOT16/train/'
    model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    detection_model_dir = '../object_detection/models/'+ model_name + '/saved_model'
    output_custom_det_dir = '../resources/detections/SSD/Custom Bboxes/MOT16_train/'

    detection_model = load_detection_model(detection_model_dir)
    generate_bboxes(mot_dir, detection_model, output_custom_det_dir)
