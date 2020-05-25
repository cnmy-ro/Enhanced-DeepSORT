#! /bin/bash

online_detection=0
min_confidence=0.4
display=0


# default max_cosine_distance is 0.2

################################################################################
max_cosine_distance=0.3
benchmark_results_path="./Results/Vehicle tracking/03cosineThreshold.txt"
echo Bechmarking all 3 detectors for cosine threshold $max_cosine_distance

for detector in "DPM" "RCNN" "SSD"
do
	python run_vehicle_tracker.py --detector $detector \
	                                --online_detection $online_detection \
	                                --min_confidence $min_confidence \
	                                --max_cosine_distance $max_cosine_distance \
	                                --display $display

	python run_detrac_benchmark.py --output_path $benchmark_results_path
done

