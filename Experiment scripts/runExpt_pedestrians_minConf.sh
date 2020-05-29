#! /bin/bash

online_detection=0
max_cosine_distance=0.2
display=0

# default min_confidence is 0.4


################################################################################
min_confidence=0.0
echo Bechmarking both detectors for confidence threshold $min_confidence 

for detector in "DPM" "SSD"
do
	python run_pedestrian_tracker.py --detector $detector \
	                                --online_detection $online_detection \
	                                --min_confidence $min_confidence \
	                                --max_cosine_distance $max_cosine_distance \
	                                --display $display	
done

python run_mot16_benchmark.py --eval_dir_dpm  "./Results/Pedestrian Tracking/EVAL_DPM/trackOutput-0.0minConf/" \
							   --eval_dir_ssd "./Results/Pedestrian Tracking/EVAL_SSD/trackOutput-0.0minConf/" \
						   --output_path "./Results/Pedestrian Tracking/benchmark_00minConf.txt"


################################################################################
min_confidence=0.2
echo Bechmarking all detectors for confidence threshold $min_confidence  

for detector in "DPM" "SSD"
do
	python run_pedestrian_tracker.py --detector $detector \
	                                --online_detection $online_detection \
	                                --min_confidence $min_confidence \
	                                --max_cosine_distance $max_cosine_distance \
	                                --display $display	
done

python run_mot16_benchmark.py --eval_dir_dpm  "./Results/Pedestrian Tracking/EVAL_DPM/trackOutput-0.2minConf/" \
							   --eval_dir_ssd "./Results/Pedestrian Tracking/EVAL_SSD/trackOutput-0.2minConf/" \
						   --output_path "./Results/Pedestrian Tracking/benchmark_02minConf.txt"


################################################################################
min_confidence=0.6
echo Bechmarking all detectors for confidence threshold $min_confidence 

for detector in "DPM" "SSD"
do
	python run_pedestrian_tracker.py --detector $detector \
	                                --online_detection $online_detection \
	                                --min_confidence $min_confidence \
	                                --max_cosine_distance $max_cosine_distance \
	                                --display $display	
done

python run_mot16_benchmark.py --eval_dir_dpm  "./Results/Pedestrian Tracking/EVAL_DPM/trackOutput-0.6minConf/" \
							   --eval_dir_ssd "./Results/Pedestrian Tracking/EVAL_SSD/trackOutput-0.6minConf/" \
							   --output_path "./Results/Pedestrian Tracking/benchmark_06minConf.txt"