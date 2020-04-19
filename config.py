###############################################################################
#                         OBJECT DETECTION
###############################################################################

#DETECTION_MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29' # ~8-9 FPS
DETECTION_MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09' # ~12 FPS
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'


###############################################################################
#                             TRACKING
###############################################################################
SEQUENCE_DIR = './MOT16/test/MOT16-06'
DETECTION_FILE = './resources/detections/MOT16_POI_test/MOT16-06.npy'
OUTPUT_FILE = '/TMP/HYPOTHESES.TXT'
MIN_CONFIDENCE = 0.3
NMS_MAX_OVERLAP = 1.0
MIN_DETECTION_HEIGHT = 0
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100
DISPLAY = True
