################################################################################
#                             GENERAL
################################################################################
# RUN_MODE options:
    #     'humans-CAM'       - Perform online detection and tracking on webcam stream
    #     'humans-EVAL'      - Perform evaluation on MOT16 data, store the results
    #     'vehicles-TEST'    - Perform vehicle tracking using test video/image sequence
    #     'vehicles-EVAL'    - Perform evaluation on UA-DETRAC data, store the results

RUN_MODE = 'vehicles-EVAL'


# EVAL mode detector options:
    #     'DPM'  - Use default pre-computed DPM-v5 detections
    #     'SSD'  - Use (pre-computed or online) MobileNetv2-SSD detections
    #     'RCNN' - Use default pre-computed RCNN detections -- only for vehicle tracking

EVAL_DETECTOR_SETTINGS = {'Detector': 'DPM',
                          'Online detection': False} # Online detection is possible only while using SSD}


################################################################################
#                         Object detection
################################################################################

#DETECTION_MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29' # ~8-9 FPS
DETECTION_MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09' # ~12 FPS
PATH_TO_LABELS = './object_detection/data/mscoco_label_map.pbtxt'



###############################################################################
#                             TRACKING
###############################################################################

MIN_CONFIDENCE = 0.4

NMS_MAX_OVERLAP = 1.0
MIN_DETECTION_HEIGHT = 0
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100
DISPLAY = False

RESULTS_DIR = './Results/'

# -----------------------------------------------------------------------------
#                          Human tracking
# -----------------------------------------------------------------------------
MOT16_TRAIN_DATA_DIR = './Data/MOT16/train/'
MOT16_DETECTION_DIR_DPM = './Resources/Humans/Detections/DPM/MOT16_POI_train/'
MOT16_DETECTION_DIR_SSD = './Resources/Humans/Detections/SSD/MOT16_POI_train/'

HUMAN_ENCODER_PATH = './Resources/Humans/human_encoder_model/mars-small128.pb'


# -----------------------------------------------------------------------------
#                         Vehicle tracking
# -----------------------------------------------------------------------------
USE_GAUSSIAN_MASK = True
VEHICLE_DATA_DIR = "./Data/UA-DETRAC/Insight-MVT_Annotation_Train/"
USE_PRECOMPUTED_DETECTIONS = True  # True: use precomputed-feature detections (.npy files) in Vehicle-EVAL mode, False: Use bboxes directly (txt files)
VEHICLE_DETECTION_DIR_BASE = "./Resources/Vehicles/Detections/"
VEHICLE_BBOXES_DIR_BASE = "./Resources/Vehicles/Bboxes/"

VEHICLE_ENCODER_PATH = './Resources/Vehicles/vehicle_encoder_model/ckpts/model640.pt'


###############################################################################
#                               MISC.
###############################################################################

# Cam mode --
#    Frame dimensions options : (640, 480), (320,240), (160,120)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480