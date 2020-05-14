################################################################################
#                             GENERAL
################################################################################
# RUN_MODE options:
    #     'humans-CAM'  - Perform online detection and tracking on webcam stream
    #     'humans-EVAL' - Perform evaluation on MOT16 data, store the results
    #     'vehicles'    - Peform vehicle tracking using saved video/image sequence

RUN_MODE = 'humans-CAM'



################################################################################
#                         Object detection
################################################################################

#DETECTION_MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29' # ~8-9 FPS
DETECTION_MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09' # ~12 FPS
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
MIN_CONFIDENCE = 0.45




###############################################################################
#                          HUMAN TRACKING
###############################################################################

# -----------------------------------------------------------------------------
#                         Detector settings
# -----------------------------------------------------------------------------

# EVAL mode detector options:
    #     'default' - Use default pre-computed (DPM-v5) detections
    #     'ssd' - Use (pre-computed or online) MobileNetv2-SSD detections
EVAL_DETECTOR_SETTINGS = {'Online detection': True, # Online detection is possible only while using SSD
                          'Detector': 'ssd'}

# -----------------------------------------------------------------------------
#                         Tracker settings
# -----------------------------------------------------------------------------
MOT16_TRAIN_DATA_DIR = './MOT16/train/'
DETECTION_DIR_DEFAULT = './resources/detections/Default/MOT16_POI_train/'
DETECTION_DIR_SSD = './resources/detections/SSD/MOT16_POI_train/'
RESULTS_DIR = './Results/'

NMS_MAX_OVERLAP = 1.0
MIN_DETECTION_HEIGHT = 0
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100
DISPLAY = False


# -----------------------------------------------------------------------------
#                              Others
# -----------------------------------------------------------------------------
# Cam mode --
#    Frame dimensions options : (640, 480), (320,240), (160,120)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480



###############################################################################
#                         VEHICLE TRACKING
###############################################################################

USE_GAUSSIAN_MASK = False