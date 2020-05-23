import os, time, logging

import numpy as np
import cv2
import torch
import torchvision

# DeepSORT imports
from deep_sort_utils import visualization as dsutil_viz
from deep_sort_utils import preprocessing as dsutil_prep
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from scipy.stats import multivariate_normal

# Custom imports
import custom_utils
from config import *


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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


def get_gaussian_mask():
    #128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5,0.5])
    sigma = np.array([0.22,0.22])
    covariance = np.diag(sigma**2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z  = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask

def pre_process(frame, bboxes):
        bboxes = np.array(bboxes)

        transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                     torchvision.transforms.Resize((128,128)),
                                                     torchvision.transforms.ToTensor()])

        crops = []
        for bbox in bboxes:
            x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            try:
                crop = frame[y:y+h, x:x+w, :]
                crop = transforms(crop)
                crops.append(crop)
            except:
                continue

        crops = torch.stack(crops)
        return crops


def create_detections(detection_mat, frame_idx, min_height=0):
    print(detection_mat.shape)
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == (frame_idx-1)

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list



def cvt_to_detection_objects(frame, bboxes, detection_scores, encoder, gaussian_mask):

    processed_crops = pre_process(frame, bboxes)
    if USE_GAUSSIAN_MASK:
        processed_crops = gaussian_mask * processed_crops

    features = encoder.forward_once(processed_crops)
    features = features.detach().cpu().numpy()

    if len(features.shape)==1:
        features = np.expand_dims(features,0)

    detection_list = []
    for  bbox, score, feature in zip(bboxes, detection_scores, features):
        feature = feature/np.linalg.norm(feature, ord=2) # Normalizing the feature vector
        detection_list.append(Detection(bbox, score, feature))

    bboxes = np.array([d.tlwh for d in detection_list])

    detection_scores = np.array([d.confidence for d in detection_list])
    indices = dsutil_prep.non_max_suppression(bboxes, NMS_MAX_OVERLAP, detection_scores)

    detection_list = [detection_list[i] for i in indices]

    return detection_list


###############################################################################

def run_test_mode(detection_model, video_path):

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE , NN_BUDGET)
    tracker= Tracker(metric)
    encoder = torch.load(VEHICLE_ENCODER_PATH, map_location='cpu')

    gaussian_mask = get_gaussian_mask()

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    while True:
        t1 = time.time()

        frame_id += 1
        ret,frame = cap.read()
        print("Frame:", frame_id)
        if ret is False: break

        frame = frame.astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Give RGB frame to object detector
        bboxes, detection_scores = detect_vehicles(detection_model, frame_rgb)

        # Give RGB frame to extract features
        detection_list = cvt_to_detection_objects(frame, bboxes, detection_scores, encoder, gaussian_mask)

        detection_list = [d for d in detection_list if d.confidence >= MIN_CONFIDENCE]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detection_list])
        scores = np.array([d.confidence for d in detection_list])
        indices = dsutil_prep.non_max_suppression(boxes, NMS_MAX_OVERLAP, scores)
        detection_list = [detection_list[i] for i in indices]

        # Update tracker
        tracker.predict()
        tracker.update(detection_list)

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
            # if not track.is_confirmed() or track.time_since_update > 0:
            #     continue
            color = dsutil_viz.create_unique_color_uchar(track.track_id)
            text_size = cv2.getTextSize(str(track.track_id), cv2.FONT_HERSHEY_PLAIN, 1, 2)
            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
            cv2.rectangle(output_img, pt1, pt2, color, -1)
            cv2.putText(output_img, str(track.track_id), center, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        fps = 1/(time.time()-t1)
        cv2.putText(output_img, 'FPS: {:.1f}'.format(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('detection output', output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2. destroyAllWindows()
    cap.release()



def run_eval_mode(detection_model):
    sequence_names = sorted(os.listdir(VEHICLE_DATA_DIR))

    encoder = torch.load(VEHICLE_ENCODER_PATH, map_location='cpu')

    if not EVAL_DETECTOR_SETTINGS['Online detection']:
        detection_dir = VEHICLE_DETECTION_DIR_BASE + EVAL_DETECTOR_SETTINGS['Detector'] + '/'
    else:
        # Add online detection capabilities
        pass
    gaussian_mask = get_gaussian_mask()

    # Print parameters before starting ----------------
    logging.info("Detections: {}".format(EVAL_DETECTOR_SETTINGS['Detector']))

    for sequence_name in sequence_names:
        sequence_dir = VEHICLE_DATA_DIR + sequence_name + '/'

        if EVAL_DETECTOR_SETTINGS['Online detection']:
            if EVAL_DETECTOR_SETTINGS['Detector'] == 'DPM':
                raise Exception(" Online detection is possible only when using SSD")
            detection_file = None
        else:
            detection_file = detection_dir + sequence_name + '.npy'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
        tracker = Tracker(metric)
        results = []

        img_file_names = sorted(os.listdir(VEHICLE_DATA_DIR+sequence_name))
        n_frames = len(img_file_names)

        logger.debug(detection_file)

        for frame_idx in range(1, n_frames+1):
            t1 = time.time()

            img_file_path = VEHICLE_DATA_DIR + sequence_name + '/' + img_file_names[frame_idx-1]
            logger.debug(img_file_path)
            frame = cv2.imread(img_file_path)


            if not EVAL_DETECTOR_SETTINGS['Online detection']: # Use pre-computed detections
                detections_list = np.load(detection_file, allow_pickle=True)
                detection_list = create_detections(detections_list, frame_idx, MIN_DETECTION_HEIGHT)

            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bboxes, detection_scores = detect_vehicles(detection_model, frame_rgb)
                detection_list = cvt_to_detection_objects(frame_rgb, bboxes, detection_scores, encoder, gaussian_mask)


            detection_list = [d for d in detection_list if d.confidence >= MIN_CONFIDENCE]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detection_list])
            scores = np.array([d.confidence for d in detection_list])
            indices = dsutil_prep.non_max_suppression(boxes, NMS_MAX_OVERLAP, scores)
            detection_list = [detection_list[i] for i in indices]

            # Update tracker.
            tracker.predict()
            tracker.update(detection_list)

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
                text_size = cv2.getTextSize(str(track.track_id), cv2.FONT_HERSHEY_PLAIN, 1, 2)
                center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
                pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
                cv2.rectangle(output_img, pt1, pt2, color, -1)
                cv2.putText(output_img, str(track.track_id), center, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            fps = 1/(time.time()-t1)
            cv2.putText(output_img, 'FPS: {:.1f}'.format(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('detection output', output_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            # Store results.
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()
                results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], fps])

        cv2. destroyAllWindows()

        # Store results.
        if not EVAL_DETECTOR_SETTINGS['Online detection']:
            output_dir = RESULTS_DIR + 'Vehicle Tracking/' + 'EVAL_' + EVAL_DETECTOR_SETTINGS['Detector'] + '/Tracking output/'
            output_file_path = output_dir + sequence_name + '.txt'
        else:
            output_file_path = '/tmp/hypotheses.txt'

        with open(output_file_path, 'w') as output_file:
            avg_fps = 0
            for row in results:
                output_file.write("{:d},{:d},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(row[0], row[1], row[2], row[3], row[4], row[5]))
                avg_fps += row[6]
        avg_fps /= n_frames
        logger.info("Average FPS: {:.2f}".format(avg_fps))


###############################################################################

if __name__ == '__main__':
    detection_model, category_index = custom_utils.load_detection_model(DETECTION_MODEL_NAME)
    # video_path = "./vdo.avi"
    # run_tracker(detection_model, video_path)
    run_eval_mode(detection_model)
