#! /bin/bash

#python track_vehicles.py --detector DPM  --online_detection 0 --min_confidence 0.4 --max_cosine_distance 0.2 --display 0

python track_vehicles.py --detector RCNN  --online_detection 0 --min_confidence 0.4 --max_cosine_distance 0.2 --display 0

python track_vehicles.py --detector SSD  --online_detection 0 --min_confidence 0.4 --max_cosine_distance 0.2 --display 0