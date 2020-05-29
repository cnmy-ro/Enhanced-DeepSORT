import argparse
import numpy as np

import motmetrics as mm


parser = argparse.ArgumentParser()
parser.add_argument('--eval_dir_dpm',type=str, required=True)
parser.add_argument('--eval_dir_ssd',type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)

args = parser.parse_args()

###############################################################################

'''
Specs:
    3 accumulators - one for the ground truth,
                     one for default DPM detection results,
                     one for SSD detection results
'''

train_sequence_names = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09',
                        'MOT16-10', 'MOT16-11', 'MOT16-13']

mot_train_dir = "./Data/MOT16/train/"

## Default paths -----------------------------------------------
# eval_dir_dpm = "./Results/Task-1/EVAL_DPM/Tracking output/"
# eval_dir_ssd = "./Results/Task-1/EVAL_SSD/Tracking output/"
# output_path = "./Results/Task-1/benchmark_results.txt"

## For experiment ----------------------------------------------
eval_dir_dpm = args.eval_dir_dpm
eval_dir_ssd = args.eval_dir_ssd
output_path = args.output_path
##

# Initialize the Py-Motmetrics accumulators
gt_accumulator = mm.MOTAccumulator(auto_id=True)
dpm_accumulator = mm.MOTAccumulator(auto_id=True)
ssd_accumulator = mm.MOTAccumulator(auto_id=True)


###############################################################################

# Run on all train sequences
print("Computing benchmark metrics ----")
for sequence_name in train_sequence_names:
    print("Processing sequence:", sequence_name)

    ground_truth_path = mot_train_dir + sequence_name + '/gt/gt.txt'
    ground_truth = np.loadtxt(ground_truth_path, delimiter=',')
    # Filter out the GT entries with flag 0
    ground_truth = ground_truth[ground_truth[:, 6]==1]
    ground_truth = ground_truth[:, :6]
    ground_truth = np.array(sorted(ground_truth, key=lambda gt: gt[0]))

    eval_result_dpm_path = eval_dir_dpm + sequence_name + '.txt'
    eval_result_dpm = np.loadtxt(eval_result_dpm_path, delimiter=',')

    eval_result_ssd_path = eval_dir_ssd + sequence_name + '.txt'
    eval_result_ssd = np.loadtxt(eval_result_ssd_path, delimiter=',')


    min_frame_idx = int(min(ground_truth[:,0]))
    max_frame_idx = int(max(ground_truth[:,0]))

    for frame_idx in range(min_frame_idx, max_frame_idx+1):
        gt_frame = ground_truth[ground_truth[:,0]==frame_idx]
        gt_frame_object_ids = gt_frame[:,1]
        gt_frame_bboxes = gt_frame[:, 2:6]

        dpm_frame = eval_result_dpm[eval_result_dpm[:,0]==frame_idx]
        #if dpm_frame.shape[0] != 0: # Only proceed if DPM result has this frame
        dpm_frame_object_ids = dpm_frame[:,1]
        dpm_frame_bboxes = dpm_frame[:,2:6]

        ssd_frame = eval_result_ssd[eval_result_ssd[:,0]==frame_idx]
        #if ssd_frame.shape[0] != 0:
        ssd_frame_object_ids = ssd_frame[:,1]
        ssd_frame_bboxes = ssd_frame[:,2:6]


        # Calculate distance  matrix (IoU)...
        self_dist_matrix = mm.distances.iou_matrix(gt_frame_bboxes, gt_frame_bboxes)
        gt_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                              gt_frame_object_ids, # Predicted objects (IDs)
                              self_dist_matrix)

        #if dpm_frame.shape[0] != 0:
        dpm_distance_matrix = mm.distances.iou_matrix(gt_frame_bboxes, dpm_frame_bboxes)
        dpm_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                               dpm_frame_object_ids, # Predicted objects (IDs)
                               dpm_distance_matrix)

        #if ssd_frame.shape[0] != 0:
        ssd_distance_matrix = mm.distances.iou_matrix(gt_frame_bboxes, ssd_frame_bboxes)
        ssd_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                               ssd_frame_object_ids, # Predicted objects (IDs)
                               ssd_distance_matrix)



mh = mm.metrics.create()
summary = mh.compute_many([gt_accumulator, dpm_accumulator, ssd_accumulator],
                           names=['Best possible scores','DPMv5', 'MobileNetv2-SSD'],
                           metrics=['mota', 'motp', 'mostly_tracked', 'mostly_lost', 'num_switches', 'num_fragmentations'])

strsummary = mm.io.render_summary(summary,
                                  formatters=mh.formatters,
                                  namemap={'mota':'MOTA', 'motp':'MOTP',
                                           'mostly_tracked':'MT', 'mostly_lost':'ML',
                                           'num_switches': 'ID', 'num_fragmentations':'FM'})

print(strsummary)

with open(output_path, 'w') as output_file:
    output_file.write(strsummary)