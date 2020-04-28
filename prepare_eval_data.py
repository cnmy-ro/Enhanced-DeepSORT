import numpy as np
import pandas as pd

'''
Temporary file:
    To be integrated into "run_mot16_eval.py"
'''

train_sequence_names = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09',
                        'MOT16-10', 'MOT16-11', 'MOT16-13']

mot_train_dir = "./MOT16/train/"

eval_dir_default = "./Results/Task-1/EVAL_default/Tracking output/"
# output_dir_default = "./Results/Task-1/EVAL_default/Benchmark results/"

eval_dir_ssd = "./Results/Task-1/EVAL_ssd/Tracking output/"
# output_dir_ssd = "./Results/Task-1/EVAL_ssd/Benchmark results/"


###############################################################################
# Trials on just one sequence

sequence_name = 'MOT16-02'

ground_truth_path = mot_train_dir + sequence_name + '/gt/gt.txt'
eval_result_default_path = eval_dir_default + sequence_name + '.txt'

ground_truth = np.loadtxt(ground_truth_path, delimiter=',')
ground_truth = ground_truth[:, :6]
ground_truth = np.array(sorted(ground_truth, key=lambda gt: gt[0]))


eval_result_default = np.loadtxt(eval_result_default_path, delimiter=',')

min_frame_idx = int(min(ground_truth[:,0]))
max_frame_idx = int(max(ground_truth[:,0]))

for frame_idx in range(min_frame_idx, max_frame_idx+1):
    gt_frame = ground_truth[ground_truth[:,0]==frame_idx]
    n_ids_gt = gt_frame.shape[0]
    gt_objects = gt_frame[:,1]

    er_def_frame = eval_result_default[eval_result_default[:,0]==frame_idx]
    n_ids_er_def = er_def_frame.shape[0]
    eval_default_objects = er_def_frame[:,1]

    dist_matrix_default = np.zeros((n_ids_gt,n_ids_er_def))
    # Calculate object distances...

