#! /bin/bash

online_detection=0
max_cosine_distance=0.2
display=0

# default min_confidence is 0.4


################################################################################
min_confidence=0.1
echo Bechmarking all detectors for confidence threshold $min_confidence 

for detector in "DPM" "RCNN" "SSD"
do
	python run_vehicle_tracker.py --detector $detector \
	                                --online_detection $online_detection \
	                                --min_confidence $min_confidence \
	                                --max_cosine_distance $max_cosine_distance \
	                                --display $display	
done

python run_detrac_benchmark.py --eval_dir_dpm  "./Results/Vehicle Tracking/EVAL_DPM/trackOutput-0.1minConf/" \
							   --eval_dir_rcnn  "./Results/Vehicle Tracking/EVAL_RCNN/trackOutput-0.1minConf/" \
							   --eval_dir_ssd "./Results/Vehicle Tracking/EVAL_SSD/trackOutput-0.1minConf/" \
						       --output_path "./Results/Vehicle Tracking/benchmark_01minConf.txt"


################################################################################
min_confidence=0.3
echo Bechmarking all detectors for confidence threshold $min_confidence  

for detector in "DPM" "RCNN" "SSD"
do
	python run_vehicle_tracker.py --detector $detector \
	                                --online_detection $online_detection \
	                                --min_confidence $min_confidence \
	                                --max_cosine_distance $max_cosine_distance \
	                                --display $display	
done

python run_detrac_benchmark.py --eval_dir_dpm  "./Results/Vehicle Tracking/EVAL_DPM/trackOutput-0.3minConf/" \
							   --eval_dir_rcnn  "./Results/Vehicle Tracking/EVAL_RCNN/trackOutput-0.3minConf/" \
							   --eval_dir_ssd "./Results/Vehicle Tracking/EVAL_SSD/trackOutput-0.3minConf/" \
						       --output_path "./Results/Vehicle Tracking/benchmark_03minConf.txt"


