
python run_detrac_benchmark.py --eval_dir_dpm  "./Results/Vehicle Tracking/EVAL_DPM/trackOutput-0.5cosineThresh/" \
							   --eval_dir_rcnn  "./Results/Vehicle Tracking/EVAL_RCNN/trackOutput-0.5cosineThresh/" \
							   --eval_dir_ssd "./Results/Vehicle Tracking/EVAL_SSD/trackOutput-0.5cosineThresh/" \
							   --output_path "./Results/Vehicle Tracking/benchmark_05cosineThreshold.txt"


###########################################################################


python run_detrac_benchmark.py --eval_dir_dpm  "./Results/Vehicle Tracking/EVAL_DPM/trackOutput-0.0minConf/" \
							   --eval_dir_rcnn  "./Results/Vehicle Tracking/EVAL_RCNN/trackOutput-0.0minConf/" \
							   --eval_dir_ssd "./Results/Vehicle Tracking/EVAL_SSD/trackOutput-0.0minConf/" \
						       --output_path "./Results/Vehicle Tracking/benchmark_00minConf.txt"


python run_detrac_benchmark.py --eval_dir_dpm  "./Results/Vehicle Tracking/EVAL_DPM/trackOutput-0.1minConf/" \
							   --eval_dir_rcnn  "./Results/Vehicle Tracking/EVAL_RCNN/trackOutput-0.1minConf/" \
							   --eval_dir_ssd "./Results/Vehicle Tracking/EVAL_SSD/trackOutput-0.1minConf/" \
						       --output_path "./Results/Vehicle Tracking/benchmark_01minConf.txt"


python run_detrac_benchmark.py --eval_dir_dpm  "./Results/Vehicle Tracking/EVAL_DPM/trackOutput-0.2minConf/" \
							   --eval_dir_rcnn  "./Results/Vehicle Tracking/EVAL_RCNN/trackOutput-0.2minConf/" \
							   --eval_dir_ssd "./Results/Vehicle Tracking/EVAL_SSD/trackOutput-0.2minConf/" \
						       --output_path "./Results/Vehicle Tracking/benchmark_02minConf.txt"


python run_detrac_benchmark.py --eval_dir_dpm  "./Results/Vehicle Tracking/EVAL_DPM/trackOutput-0.3minConf/" \
							   --eval_dir_rcnn  "./Results/Vehicle Tracking/EVAL_RCNN/trackOutput-0.3minConf/" \
							   --eval_dir_ssd "./Results/Vehicle Tracking/EVAL_SSD/trackOutput-0.3minConf/" \
						       --output_path "./Results/Vehicle Tracking/benchmark_03minConf.txt"