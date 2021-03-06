'''

Main code file for DeepSORT pedestrian tracker.

Arguments:
    --mode : 'cam' or 'eval'
    --detector : 'DPM' or 'SSD'
    --online_detection : 'True' for running SSD on the fly
    --min_confidence : 0-1 float
    --max_cosine_distance : 0-1 float
    --display: 'True' to visualize tracker output in 'eval' mode

'''

import os, time, logging, argparse

import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


# TF object detection imports
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# DeepSORT imports
from deep_sort_utils import preprocessing as dsutil_prep
from deep_sort_utils import visualization as dsutil_viz
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from Tools import generate_mot16_detections as gen_det

# Custom imports
import custom_utils
from config import *


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def detect_humans(detection_model, frame):
    '''

    Wrapper function for human detection using the given SSD model
    Returns:
        detection_list - list of bboxes in TLWH format
        detction_scores

    '''
    output_dict =  custom_utils.run_inference_for_single_image(detection_model, frame)

    all_bboxes = output_dict['detection_boxes']
    all_detection_classes = output_dict['detection_classes']
    all_detection_scores = output_dict['detection_scores']

    bboxes = [] # List of TLWH bboxes
    detection_scores = [] # List of score for all humans detected
    for i, box in enumerate(all_bboxes):
        det_class = all_detection_classes[i]
        det_score = all_detection_scores[i]
        if det_class == 1: # If the detected object is a person
            y1,x1,y2,x2 = int(box[0]*frame.shape[0]), int(box[1]*frame.shape[1]), int(box[2]*frame.shape[0]), int(box[3]*frame.shape[1])
            bboxes.append(tuple([x1,y1,x2-x1,y2-y1])) # TLWH format bboxes
            detection_scores.append(det_score)

    return bboxes, detection_scores


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


def cvt_to_detection_objects(frame, bboxes, confidence_scores, encoder):
    detection_list = []
    features = encoder(frame, bboxes)
    for i, bbox in enumerate(bboxes):
        detection_list.append(Detection(bbox, confidence_scores[i], features[i]))
    return detection_list



###############################################################################
#                               Run options
###############################################################################
def run_cam_mode(detection_model, args):

    '''

    Run the tracker on camera stream

    '''

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance, NN_BUDGET)
    tracker = Tracker(metric)

    encoder = gen_det.create_box_encoder(HUMAN_ENCODER_PATH, batch_size=32)

    # Cam loop
    while True:
      ret, frame = cam.read()
      t1 = time.time()

      # Detect humans in the frame
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      bboxes, detection_scores = detect_humans(detection_model, frame_rgb)
      detection_list = cvt_to_detection_objects(frame_rgb, bboxes, detection_scores, encoder)

      # Run non-maxima suppression.
      detection_list = [d for d in detection_list if d.confidence >= args.min_confidence]
      boxes = np.array([d.tlwh for d in detection_list])
      scores = np.array([d.confidence for d in detection_list])
      indices = dsutil_prep.non_max_suppression(boxes, NMS_MAX_OVERLAP, scores)
      detection_list = [detection_list[i] for i in indices]

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
          color = dsutil_viz.create_unique_color_uchar(track.track_id)
          x,y,w,h = track.to_tlwh()
          pt1 = int(x), int(y)
          pt2 = int(x + w), int(y + h)
          cv2.rectangle(output_img, pt1, pt2, color, 2)
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



def run_eval_mode(detection_model, args):
    '''

    Evaluate the tracker on MOT-16 training set

    '''

    train_sequence_names = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09',
                            'MOT16-10', 'MOT16-11', 'MOT16-13']

    # Print parameters before starting ----------------
    logger.info("\nConfig ---- \n----Detector: {} \n----Online detection: {}".format(args.detector, args.online_detection))


    for sequence_name in train_sequence_names:
        logger.info("Processing sequence: {}".format(sequence_name))
        sequence_dir = MOT16_TRAIN_DATA_DIR + sequence_name + '/'

        if args.online_detection == 1:
            if args.detector  == 'DPM':
                raise Exception(" Online detection is possible only when using SSD")
            detection_file = None
        else:
            detection_dir = MOT16_DETECTION_DIR_BASE + args.detector + "/MOT16_POI_train/"
            detection_file = detection_dir + sequence_name + '.npy'
            detections = np.load(detection_file, allow_pickle=True)
            #print("detections shape:", detections.shape)

        seq_info = gather_sequence_info(sequence_dir, detection_file)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", args.max_cosine_distance, NN_BUDGET)
        tracker = Tracker(metric)
        results = []

        encoder = gen_det.create_box_encoder(HUMAN_ENCODER_PATH, batch_size=32)

        n_frames = len(seq_info['image_filenames'])
        logger.info("----Total frames:{}".format(n_frames))
        logger.info("----Detections file:{}".format(detection_file))



        def frame_callback(vis, frame_idx):
            logger.info("Processing Sequence {}, Frame {:05d}" .format(sequence_name, frame_idx))

            frame = cv2.imread(seq_info['image_filenames'][frame_idx])
            t1 = time.time()

            if args.online_detection == 0: # If not online detection- Use pre-computed detections
                # Load image and generate detections.
                detection_list = create_detections(detections, frame_idx, MIN_DETECTION_HEIGHT)
                detection_list = [d for d in detection_list if d.confidence >= args.min_confidence]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detection_list])
                scores = np.array([d.confidence for d in detection_list])
                indices = dsutil_prep.non_max_suppression(boxes, NMS_MAX_OVERLAP, scores)
                detection_list = [detection_list[i] for i in indices]

            else: # Use Mobilenet-SSD on the fly
                # Detect humans
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bboxes, detection_scores = detect_humans(detection_model, frame_rgb)
                detection_list = cvt_to_detection_objects(frame, bboxes, detection_scores, encoder)

            # Update tracker.
            tracker.predict()
            tracker.update(detection_list)

            # FPS counter
            fps = 1/(time.time()-t1)

            # Update visualization.
            if args.display == 1:
                vis.set_image(frame.copy())
                vis.draw_detections(detection_list)
                vis.draw_trackers(tracker.tracks)

            # Store results.
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()
                results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], fps])

        # Visualize
        if args.display == 1:
            visualizer = dsutil_viz.Visualization(seq_info, update_ms=5)
        else:
            visualizer = dsutil_viz.NoVisualization(seq_info)
        visualizer.run(frame_callback)

        # Store results.

        if args.online_detection == 0: # If not online detection
            output_dir = RESULTS_DIR + 'Pedestrian Tracking/' + 'EVAL_' + args.detector + '/trackOutput-{}minConf/'.format(args.min_confidence)
            output_file_path = output_dir + sequence_name + '.txt'
        else:
            output_file_path = './Results/temp_hypotheses.txt'


        with open(output_file_path, 'w') as output_file:
            avg_fps = 0
            for row in results:
                output_file.write("{:d},{:d},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(row[0], row[1], row[2], row[3], row[4], row[5]))
                avg_fps += row[6]
        avg_fps /= n_frames
        #logger.info("----Average FPS: {:.2f}".format(avg_fps))


#######################################################################################

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help="'cam': Run on webcam stream; 'eval': Evaluate on MOT16 benchmark",
                        default='eval')

    parser.add_argument('--detector', type=str, help="Options: DPM, SSD",
                        default=EVAL_DETECTOR_SETTINGS['Detector'])

    parser.add_argument('--online_detection', type=int, help="0: Use pre-computed detections, 1: Online detection",
                        default=int(EVAL_DETECTOR_SETTINGS['Online detection']))

    parser.add_argument('--min_confidence', type=float, help="Minimum detection confidence [0-1]",
                        default=MIN_CONFIDENCE)

    parser.add_argument('--max_cosine_distance', type=float, help="Max. cosine distance threshold for features",
                        default=MAX_COSINE_DISTANCE)

    parser.add_argument('--display', type=int, help="Display the tracking results",
                        default=int(DISPLAY))

    args = parser.parse_args()
    return args

#######################################################################################

if __name__ == '__main__':

    args = parse_args()

    detection_model, category_index = custom_utils.load_detection_model(DETECTION_MODEL_NAME)
    logger.debug(category_index[1]) # Human

    if args.mode == 'cam':
        run_cam_mode(detection_model, args)

    if args.mode == 'eval':
        run_eval_mode(detection_model, args)
