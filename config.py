###############################################################################
#                         OBJECT DETECTION
###############################################################################

#DETECTION_MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29' # ~8-9 FPS
DETECTION_MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09' # ~12 FPS
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'


###############################################################################
#                             Tracking
###############################################################################
sequence_dir = './MOT16/test/MOT16-06'
detection_file = './resources/detections/MOT16_POI_test/MOT16-06.npy'
output_file = '/tmp/hypotheses.txt'
min_confidence = 0.3
nms_max_overlap = 1.0
min_detection_height = 0
max_cosine_distance = 0.2
nn_budget = 100
display = True