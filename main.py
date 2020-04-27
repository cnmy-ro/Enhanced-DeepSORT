import os, time, logging

import numpy as np
import cv2
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gen_det

from config import *

###############################################################################
'''

Check TODO.txt

'''
###############################################################################

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

###############################################################################
#                           Object detection
###############################################################################
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_detection_model(model_name):
    model_dir = 'object_detection/models/'+ model_name + '/saved_model'
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    # List of the strings that is used to add correct label for each box.
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    return model, category_index


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


def detect_humans(detection_model, frame):
    output_dict =  run_inference_for_single_image(detection_model, frame)

    # output_img = vis_util.visualize_boxes_and_labels_on_image_array(frame,
    #                                                             output_dict['detection_boxes'],
    #                                                             output_dict['detection_classes'],
    #                                                             output_dict['detection_scores'],
    #                                                             category_index,
    #                                                             instance_masks=output_dict.get('detection_masks_reframed', None),
    #                                                             use_normalized_coordinates=True,
    #                                                             line_thickness=8)


    all_bboxes = output_dict['detection_boxes']
    all_detection_classes = output_dict['detection_classes']
    all_detection_scores = output_dict['detection_scores']

    detection_list = [] # List of DeepSORT's Detection objects
    detection_scores = [] # List of score for all humans detected
    for i, box in enumerate(all_bboxes):
        det_class = all_detection_classes[i]
        det_score = all_detection_scores[i]
        if det_class == 1 and det_score > MIN_CONFIDENCE: # If the detected object is a person with a confidence of at least MIN_CONFIDENCE
            y1,x1,y2,x2 = int(box[0]*frame.shape[0]), int(box[1]*frame.shape[1]), int(box[2]*frame.shape[0]), int(box[3]*frame.shape[1])
            detection_list.append(tuple([x1,y1,x2-x1,y2-y1])) # tlwh format bboxes
            detection_scores.append(det_score)

    return detection_list, detection_scores

###############################################################################
#                                Tracking
###############################################################################

def gather_sequence_info(sequence_dir, detection_file):
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
                       for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {"sequence_name": os.path.basename(sequence_dir),
                "image_filenames": image_filenames,
                "detections": detections,
                "groundtruth": groundtruth,
                "image_size": image_size,
                "min_frame_idx": min_frame_idx,
                "max_frame_idx": max_frame_idx,
                "feature_dim": feature_dim,
                "update_ms": update_ms
                }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def cvt_to_detection_object_list(frame, bboxes, confidence_scores, encoder):
    detection_list = []
    features = encoder(frame, bboxes)
    for i, bbox in enumerate(bboxes):
        detection_list.append(Detection(bbox, confidence_scores[i], features[i]))
    return detection_list

###############################################################################
#                               Run options
###############################################################################
def run_cam_mode(detection_model):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric)

    encoder = gen_det.create_box_encoder('resources/networks/mars-small128.pb', batch_size=32)

    # Cam loop
    while True:
      ret, frame = cam.read()
      t1 = time.time()

      # Detect humans in the frame
      detection_list, detection_scores = detect_humans(detection_model, frame)
      detection_list = cvt_to_detection_object_list(frame, detection_list, detection_scores, encoder)

      # Update tracker
      tracker.predict()
      tracker.update(detection_list)

      # FPS counter
      fps = 1/(time.time()-t1)

      # Update visualization
      output_img = frame.copy()

      ## Visualize all detections
      for i, detection in enumerate(detection_list):
          x,y,w,h = detection.tlwh
          pt1 = int(x), int(y)
          pt2 = int(x + w), int(y + h)
          cv2.rectangle(output_img, pt1, pt2, (0, 0, 255), 2)

      ## Visualize confirmed tracks
      tracks = tracker.tracks
      for track in tracks:
          if not track.is_confirmed() or track.time_since_update > 0:
              continue
          color = visualization.create_unique_color_uchar(track.track_id)
          text_size = cv2.getTextSize(str(track.track_id), cv2.FONT_HERSHEY_PLAIN, 1, 2)
          center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
          pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
          cv2.rectangle(output_img, pt1, pt2, color, -1)
          cv2.putText(output_img, str(track.track_id), center, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

      cv2.putText(output_img, 'FPS: {:.1f}'.format(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
      cv2.imshow('detection output', output_img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2. destroyAllWindows()
    cam.release()



def run_eval_mode(detection_model, eval_detector_settings):
    train_sequence_names = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09',
                            'MOT16-10', 'MOT16-11', 'MOT16-13']

    if not eval_detector_settings['Online detection']:
        if eval_detector_settings['Detector'] == 'default':
            detection_dir = DETECTION_DIR_DEFAULT
        elif eval_detector_settings['Detector'] == 'ssd':
            detection_dir = DETECTION_DIR_SSD


    for sequence_name in train_sequence_names:
        sequence_dir = MOT16_TRAIN_DATA_DIR + sequence_name + '/'
        detection_file = detection_dir + sequence_name + '.npy'

        if eval_detector_settings['Online detection']:
            if eval_detector_settings['Detector'] == 'default':
                raise Exception(" Online detection is possible only when using SSD")
            detection_file = None

        seq_info = gather_sequence_info(sequence_dir, detection_file)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
        tracker = Tracker(metric)
        results = []

        encoder = gen_det.create_box_encoder('resources/networks/mars-small128.pb', batch_size=32)

        n_frames = len(seq_info['image_filenames'])

        logger.debug(detection_file)

        def frame_callback(vis, frame_idx):
            logger.info("Processing Sequence {}, Frame {:05d}" .format(sequence_name, frame_idx))

            frame = cv2.imread(seq_info['image_filenames'][frame_idx])
            t1 = time.time()

            if not eval_detector_settings['Online detection']: # Use pre-computed detections
                # Load image and generate detections.
                detection_list = create_detections(seq_info["detections"], frame_idx, MIN_DETECTION_HEIGHT)
                detection_list = [d for d in detection_list if d.confidence >= MIN_CONFIDENCE]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detection_list])
                scores = np.array([d.confidence for d in detection_list])
                indices = preprocessing.non_max_suppression(boxes, NMS_MAX_OVERLAP, scores)
                detection_list = [detection_list[i] for i in indices]

            else: # Use Mobilenet-SSD on the fly
                # Detect humans
                detection_list, detection_scores = detect_humans(detection_model, frame)
                detection_list = cvt_to_detection_object_list(frame, detection_list, detection_scores, encoder)

            # Update tracker.
            tracker.predict()
            tracker.update(detection_list)

            # FPS counter
            fps = 1/(time.time()-t1)

            # Update visualization.
            if DISPLAY:
                vis.set_image(frame.copy())
                vis.draw_detections(detection_list)
                vis.draw_trackers(tracker.tracks)

            # Store results.
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()
                results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], fps])

        # Run tracker.
        if DISPLAY:
            visualizer = visualization.Visualization(seq_info, update_ms=5)
        else:
            visualizer = visualization.NoVisualization(seq_info)
        visualizer.run(frame_callback)

        # Store results.
        if not eval_detector_settings['Online detection']:
            output_dir = RESULTS_DIR + 'Task-1/' + 'EVAL_' + eval_detector_settings['Detector'] + '/Tracking output/'
            output_file_path = output_dir + sequence_name + '.txt'
        else:
            output_file_path = '/temp/hypotheses.txt'

        with open(output_file_path, 'w') as output_file:
            avg_fps = 0
            for row in results:
                output_file.write("{:d},{:d},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(row[0], row[1], row[2], row[3], row[4], row[5]))
                avg_fps += row[6]
        avg_fps /= n_frames
        logger.info("Average FPS: {:.2f}".format(avg_fps))

###############################################################################
def run(run_mode, eval_detector_settings):
    detection_model, category_index = load_detection_model(DETECTION_MODEL_NAME)
    logger.debug(category_index[1])

    logger.debug(detection_model.inputs)
    logger.debug(detection_model.output_dtypes)

    if run_mode == 'CAM':
        run_cam_mode(detection_model)
    elif run_mode == 'EVAL':
        run_eval_mode(detection_model, eval_detector_settings)

###############################################################################
if __name__ == '__main__':

    # Run mode options:
    #     'CAM'  - Perform online detection and tracking on webcam stream
    #     'EVAL' - Perform evaluation on MOT16 data, store the results
    run_mode = 'CAM'

    # EVAL mode detector options:
    #     'default' - Use pre-computed default detections
    #     'ssd' - Use (pre-computed or online) MobileNetv2-SSD detections
    eval_detector_settings = {'Online detection': False, # Online detection is possible only while using SSD
                              'Detector': 'default'}

    run(run_mode, eval_detector_settings)