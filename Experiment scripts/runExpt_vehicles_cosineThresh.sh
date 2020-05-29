#! /bin/bash

online_detection=0
min_confidence=0.4
display=0


# default max_cosine_distance is 0.2

################################################################################
max_cosine_distance=0.3
echo Bechmarking all detectors for cosine threshold $max_cosine_distance 

for detector in "DPM" "RCNN" "SSD"
do
	python run_vehicle_tracker.py --detector $detector \
	                                --online_detection $online_detection \
	                                --min_confidence $min_confidence \
	                                --max_cosine_distance $max_cosine_distance \
	                                --display $display	
done

python run_detrac_benchmark.py --eval_dir_dpm  "./Results/Vehicle Tracking/EVAL_DPM/trackOutput-0.3cosineThresh/" \
							   --eval_dir_rcnn  "./Results/Vehicle Tracking/EVAL_RCNN/trackOutput-0.3cosineThresh/" \
							   --eval_dir_ssd "./Results/Vehicle Tracking/EVAL_SSD/trackOutput-0.3cosineThresh/" \
							   --output_path "./Results/Vehicle Tracking/benchmark_03cosineThresh.txt"


################################################################################
max_cosine_distance=0.4
echo Bechmarking all detectors for cosine threshold $max_cosine_distance 

for detector in "DPM" "RCNN" "SSD"
do
	python run_vehicle_tracker.py --detector $detector \
	                                --online_detection $online_detection \
	                                --min_confidence $min_confidence \
	                                --max_cosine_distance $max_cosine_distance \
	                                --display $display	
done

python run_detrac_benchmark.py --eval_dir_dpm  "./Results/Vehicle Tracking/EVAL_DPM/trackOutput-0.4cosineThresh/" \
							   --eval_dir_rcnn  "./Results/Vehicle Tracking/EVAL_RCNN/trackOutput-0.4cosineThresh/" \
							   --eval_dir_ssd "./Results/Vehicle Tracking/EVAL_SSD/trackOutput-0.4cosineThresh/" \
							   --output_path "./Results/Vehicle Tracking/benchmark_04cosineThresh.txt"


################################################################################
max_cosine_distance=0.5
echo Bechmarking all detectors for cosine threshold $max_cosine_distance 

for detector in "DPM" "RCNN" "SSD"
do
	python run_vehicle_tracker.py --detector $detector \
	                                --online_detection $online_detection \
	                                --min_confidence $min_confidence \
	                                --max_cosine_distance $max_cosine_distance \
	                                --display $display	
done

python run_detrac_benchmark.py --eval_dir_dpm  "./Results/Vehicle Tracking/EVAL_DPM/trackOutput-0.5cosineThresh/" \
							   --eval_dir_rcnn  "./Results/Vehicle Tracking/EVAL_RCNN/trackOutput-0.5cosineThresh/" \
							   --eval_dir_ssd "./Results/Vehicle Tracking/EVAL_SSD/trackhOutput-0.5cosineThresh/" \
							   --output_path "./Results/Vehicle Tracking/benchmark_05cosineThreshold.txt"
