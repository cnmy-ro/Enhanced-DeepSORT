import numpy as np

import motmetrics as mm

'''
Specs:
    2 accumulators - one for default detection results, one for SSD
'''

train_sequence_names = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09',
                        'MOT16-10', 'MOT16-11', 'MOT16-13']

mot_train_dir = "./MOT16/train/"

eval_dir_default = "./Results/Task-1/EVAL_default/Tracking output/"
# output_dir_default = "./Results/Task-1/EVAL_default/Benchmark results/"

eval_dir_ssd = "./Results/Task-1/EVAL_ssd/Tracking output/"
# output_dir_ssd = "./Results/Task-1/EVAL_ssd/Benchmark results/"

gt_accumulator = mm.MOTAccumulator(auto_id=False)
default_accumulator = mm.MOTAccumulator(auto_id=False)
ssd_accumulator = mm.MOTAccumulator(auto_id=False)


###############################################################################
# Trials on just one sequence

sequence_name = 'MOT16-02'

ground_truth_path = mot_train_dir + sequence_name + '/gt/gt.txt'
ground_truth = np.loadtxt(ground_truth_path, delimiter=',')
ground_truth = ground_truth[ground_truth[:, 6]==1]
ground_truth = ground_truth[:, :6]
ground_truth = np.array(sorted(ground_truth, key=lambda gt: gt[0]))

eval_result_default_path = eval_dir_default + sequence_name + '.txt'
eval_result_default = np.loadtxt(eval_result_default_path, delimiter=',')

eval_result_ssd_path = eval_dir_ssd + sequence_name + '.txt'
eval_result_ssd = np.loadtxt(eval_result_ssd_path, delimiter=',')


min_frame_idx = int(min(ground_truth[:,0]))
max_frame_idx = int(max(ground_truth[:,0]))

for frame_idx in range(min_frame_idx, max_frame_idx+1):
    gt_frame = ground_truth[ground_truth[:,0]==frame_idx]
    gt_frame_object_ids = gt_frame[:,1]
    gt_frame_bboxes = gt_frame[:, 2:6]

    default_frame = eval_result_default[eval_result_default[:,0]==frame_idx]
    default_frame_object_ids = default_frame[:,1]
    default_frame_bboxes = default_frame[:,2:6]

    ssd_frame = eval_result_ssd[eval_result_ssd[:,0]==frame_idx]
    ssd_frame_object_ids = ssd_frame[:,1]
    ssd_frame_bboxes = ssd_frame[:,2:6]


    # Calculate distance  matrix (IoU)...
    self_dist_matrix = mm.distances.iou_matrix(gt_frame_bboxes, gt_frame_bboxes)
    gt_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                          gt_frame_object_ids, # Predicted objects (IDs)
                          self_dist_matrix,
                          frameid=frame_idx)

    default_distance_matrix = mm.distances.iou_matrix(gt_frame_bboxes, default_frame_bboxes)
    default_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                           default_frame_object_ids, # Predicted objects (IDs)
                           default_distance_matrix,
                           frameid=frame_idx)

    ssd_distance_matrix = mm.distances.iou_matrix(gt_frame_bboxes, ssd_frame_bboxes)
    ssd_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                           ssd_frame_object_ids, # Predicted objects (IDs)
                           ssd_distance_matrix,
                           frameid=frame_idx)



mh = mm.metrics.create()
summary = mh.compute_many([gt_accumulator, default_accumulator, ssd_accumulator],
                          metrics=mm.metrics.motchallenge_metrics,
                          names=['gt_self','default', 'ssd'])
strsummary = mm.io.render_summary(summary,
                                  formatters=mh.formatters,
                                  namemap=mm.io.motchallenge_metric_names)
print(strsummary)
