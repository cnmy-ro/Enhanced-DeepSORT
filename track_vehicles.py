import os, time, logging

import numpy as np
import cv2
import torch
import torchvision

# DeepSORT imports
from application_util import visualization as dsutil_viz
from application_util import preprocessing as dsutil_prep
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gen_det

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
        # If the detected object is a vehicle with a confidence of at least MIN_CONFIDENCE
        if (det_class == 3 or det_class == 6 or det_class == 8) and det_score > MIN_CONFIDENCE:
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
            x,y,w,h = bbox

            try:
                crop = frame[y:y+h, x:x+w, :]
                crop = transforms(crop)
                crops.append(crop)
            except:
                continue

        crops = torch.stack(crops)
        return crops


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
        detection_list.append(Detection(bbox, score, feature))

    bboxes = np.array([d.tlwh for d in detection_list])

    detection_scores = np.array([d.confidence for d in detection_list])
    indices = dsutil_prep.non_max_suppression(bboxes, NMS_MAX_OVERLAP, detection_scores)

    detection_list = [detection_list[i] for i in indices]

    return detection_list


###############################################################################

def run_tracker(detection_model, video_path):

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE , NN_BUDGET)
    tracker= Tracker(metric)
    encoder = torch.load('./vehicle_encoder_model/ckpts/model640.pt', map_location='cpu')

    gaussian_mask = get_gaussian_mask()

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    while True:
        t1 = time.time()

        ret,frame = cap.read()
        print("Frame:", frame_id)
        if ret is False: break

        frame = frame.astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Give RGB frame to object detector
        bboxes, detection_scores = detect_vehicles(detection_model, frame_rgb)

        # Give RGB frame to extract features
        detection_list = cvt_to_detection_objects(frame_rgb, bboxes, detection_scores, encoder, gaussian_mask)

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
    cv2. destroyAllWindows()
    cap.release()


###############################################################################

if __name__ == '__main__':
    detection_model, category_index = custom_utils.load_detection_model(DETECTION_MODEL_NAME)
    video_path = "./vdo.avi"
    run_tracker(detection_model, video_path)

