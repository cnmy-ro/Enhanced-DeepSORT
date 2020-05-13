import logging

# Custom imports
import custom_utils
import track_humans
from config import *


###############################################################################
'''

Check TODO.txt

'''
###############################################################################


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run():
    # Get SSD detector
    detection_model, category_index = custom_utils.load_detection_model(DETECTION_MODEL_NAME)
    logger.debug(category_index[1])

    logger.debug(detection_model.inputs)
    logger.debug(detection_model.output_dtypes)

    if RUN_MODE == 'humans-CAM':
        track_humans.run_cam_mode(detection_model)
    elif RUN_MODE == 'humans-EVAL':
        track_humans.run_eval_mode(detection_model, EVAL_DETECTOR_SETTINGS)

    elif RUN_MODE == 'vehicles':
        track_vehicles.run_tracker(detection_model)

###############################################################################

if __name__ == '__main__':
    run()