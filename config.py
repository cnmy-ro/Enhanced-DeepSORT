################################################################################
#                             GENERAL
################################################################################
# RUN_MODE options:
    #     'humans-CAM'       - Perform online detection and tracking on webcam stream
    #     'humans-EVAL'      - Perform evaluation on MOT16 data, store the results
    #     'vehicles-TEST'    - Perform vehicle tracking using test video/image sequence
    #     'vehicles-EVAL'    - Perform evaluation on UA-DETRAC data, store the results

RUN_MODE = 'humans-CAM'


# EVAL mode detector options:
    #     'DPM' - Use default pre-computed (DPM-v5) detections
    #     'SSD' - Use (pre-computed or online) MobileNetv2-SSD detections

EVAL_DETECTOR_SETTINGS = {'Detector': 'DPM',
                          'Online detection': False} # Online detection is possible only while using SSD}


################################################################################
#                         Object detection
################################################################################

#DETECTION_MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29' # ~8-9 FPS
DETECTION_MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09' # ~12 FPS
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
MIN_CONFIDENCE = 0.45




###############################################################################
#                             TRACKING
###############################################################################

NMS_MAX_OVERLAP = 1.0
MIN_DETECTION_HEIGHT = 0
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100
DISPLAY = True

RESULTS_DIR = './Results/'

# -----------------------------------------------------------------------------
#                          Human tracking
# -----------------------------------------------------------------------------
MOT16_TRAIN_DATA_DIR = './MOT16/train/'
MOT16_DETECTION_DIR_DPM = './resources/detections/DPM/MOT16_POI_train/'
MOT16_DETECTION_DIR_SSD = './resources/detections/SSD/MOT16_POI_train/'




# -----------------------------------------------------------------------------
#                         Vehicle tracking
# -----------------------------------------------------------------------------
USE_GAUSSIAN_MASK = False
VEHICLE_DATA_DIR = "UA-DETRAC/Insight-MVT_Annotation_Train/"
VEHICLE_DETECTION_DIR_DPM = "UA-DETRAC/Object Data/Detections/DPM/"
VEHICLE_DETECTION_DIR_SSD = "UA-DETRAC/Object Data/Detections/SSD/"




###############################################################################
#                               MISC.
###############################################################################

# Cam mode --
#    Frame dimensions options : (640, 480), (320,240), (160,120)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480