import numpy as np

import motmetrics as mm

'''
Specs:
    3 accumulators - one for the ground truth,
                     one for default DPM detection results,
                     one for SSD detection results
'''

train_sequence_names = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09',
                        'MOT16-10', 'MOT16-11', 'MOT16-13']

mot_train_dir = ".Data/MOT16/train/"

eval_dir_dpm = "./Results/Vehicle Tracking/EVAL_DPM/Tracking output/"
eval_dir_rcnn = "./Results/Vehicle Tracking/EVAL_RCNN/Tracking output/"
eval_dir_ssd = "./Results/Vehicle Tracking/EVAL_SSD/Tracking output/"

output_file_path = "./Results/Vehicle Tracking/benchmark_results.txt"

# Initialize the Py-Motmetrics accumulators
gt_accumulator = mm.MOTAccumulator(auto_id=True)
dpm_accumulator = mm.MOTAccumulator(auto_id=True)
ssd_accumulator = mm.MOTAccumulator(auto_id=True)


###############################################################################

# Run on all train sequences
for sequence_name in train_sequence_names:

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

        default_frame = eval_result_dpm[eval_result_dpm[:,0]==frame_idx]
        default_frame_object_ids = default_frame[:,1]
        default_frame_bboxes = default_frame[:,2:6]

        ssd_frame = eval_result_ssd[eval_result_ssd[:,0]==frame_idx]
        ssd_frame_object_ids = ssd_frame[:,1]
        ssd_frame_bboxes = ssd_frame[:,2:6]


        # Calculate distance  matrix (IoU)...
        self_dist_matrix = mm.distances.iou_matrix(gt_frame_bboxes, gt_frame_bboxes)
        gt_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                              gt_frame_object_ids, # Predicted objects (IDs)
                              self_dist_matrix)

        default_distance_matrix = mm.distances.iou_matrix(gt_frame_bboxes, default_frame_bboxes)
        dpm_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                               default_frame_object_ids, # Predicted objects (IDs)
                               default_distance_matrix)

        ssd_distance_matrix = mm.distances.iou_matrix(gt_frame_bboxes, ssd_frame_bboxes)
        ssd_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                               ssd_frame_object_ids, # Predicted objects (IDs)
                               ssd_distance_matrix)



mh = mm.metrics.create()
summary = mh.compute_many([gt_accumulator, dpm_accumulator, ssd_accumulator],
                          metrics=mm.metrics.motchallenge_metrics,
                          names=['Perfect scores (GT v/s GT)','Default detecions (dpm-v5)', 'MobileNetv2-ssd'])
strsummary = mm.io.render_summary(summary,
                                  formatters=mh.formatters,
                                  namemap=mm.io.motchallenge_metric_names)
print(strsummary)

with open(output_file_path, 'w') as output_file:
    output_file.write(strsummary)