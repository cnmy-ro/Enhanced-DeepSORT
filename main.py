import logging

# Custom imports
import custom_utils
import track_humans, track_vehicles
from config import *


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run():
    # Get SSD detector
    detection_model, category_index = custom_utils.load_detection_model(DETECTION_MODEL_NAME)
    logger.debug(category_index[1]) # Human
    logger.debug(category_index[3]) # Car
    logger.debug(category_index[6]) # Bus
    logger.debug(category_index[8]) # Truck

    logger.debug(detection_model.inputs)
    logger.debug(detection_model.output_dtypes)

    if RUN_MODE == 'humans-CAM':
        track_humans.run_cam_mode(detection_model)
    elif RUN_MODE == 'humans-EVAL':
        track_humans.run_eval_mode(detection_model)

    elif RUN_MODE == 'vehicles-TEST':
        video_path = "./vehicle_tracking/vdo.avi"
        track_vehicles.run_tracker(detection_model, video_path)

###############################################################################

if __name__ == '__main__':
    run()