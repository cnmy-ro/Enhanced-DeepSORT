import os, argparse
import numpy as np

import motmetrics as mm


parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str,
                    default="./Results/Vehicle Tracking/benchmark_results.txt")

args = parser.parse_args()

###############################################################################

'''
Specs:
    4 accumulators - one for the ground truth,
                     one for DPM detection results,
                     one for RCNN detection results,
                     one for SSD detection results
'''
vehicle_data_dir = "./Data/UA-DETRAC/Insight-MVT_Annotation_Train/"
sequence_name_list = sorted(os.listdir(vehicle_data_dir))[:20] # First 20 sequences
ground_truth_dir = "./Data/UA-DETRAC/ground_truths/"

eval_dir_dpm = "./Results/Vehicle Tracking/EVAL_DPM/Tracking output/"
eval_dir_rcnn = "./Results/Vehicle Tracking/EVAL_RCNN/Tracking output/"
eval_dir_ssd = "./Results/Vehicle Tracking/EVAL_SSD/Tracking output/"


# Initialize the Py-Motmetrics accumulators
gt_accumulator = mm.MOTAccumulator(auto_id=True)
dpm_accumulator = mm.MOTAccumulator(auto_id=True)
rcnn_accumulator = mm.MOTAccumulator(auto_id=True)
ssd_accumulator = mm.MOTAccumulator(auto_id=True)


###############################################################################

# Run on all train sequences
for sequence_name in sequence_name_list:

    print("Processing sequence:", sequence_name)

    ground_truth_path = ground_truth_dir + sequence_name + '.txt'
    ground_truth = np.loadtxt(ground_truth_path, delimiter=',')

    eval_result_dpm_path = eval_dir_dpm + sequence_name + '.txt'
    eval_result_dpm = np.loadtxt(eval_result_dpm_path, delimiter=',')

    eval_result_rcnn_path = eval_dir_rcnn + sequence_name + '.txt'
    eval_result_rcnn = np.loadtxt(eval_result_rcnn_path, delimiter=',')

    eval_result_ssd_path = eval_dir_ssd + sequence_name + '.txt'
    eval_result_ssd = np.loadtxt(eval_result_ssd_path, delimiter=',')


    min_frame_idx = int(min(ground_truth[:,0]))
    max_frame_idx = int(max(ground_truth[:,0]))

    for frame_idx in range(min_frame_idx, max_frame_idx+1):

        #print("Frame:", frame_idx)

        gt_frame = ground_truth[ground_truth[:,0]==frame_idx]
        gt_frame_object_ids = gt_frame[:,1]
        gt_frame_bboxes = gt_frame[:, 2:6]

        dpm_frame = eval_result_dpm[eval_result_dpm[:,0]==frame_idx]
        if dpm_frame.shape[0] != 0:
          dpm_frame_object_ids = dpm_frame[:,1]
          dpm_frame_bboxes = dpm_frame[:,2:6]

        rcnn_frame = eval_result_rcnn[eval_result_rcnn[:,0]==frame_idx]
        if rcnn_frame.shape[0] != 0:
          rcnn_frame_object_ids = rcnn_frame[:,1]
          rcnn_frame_bboxes = rcnn_frame[:,2:6]

        ssd_frame = eval_result_ssd[eval_result_ssd[:,0]==frame_idx]
        if ssd_frame.shape[0] != 0:
          ssd_frame_object_ids = ssd_frame[:,1]
          ssd_frame_bboxes = ssd_frame[:,2:6]


        # Calculate distance  matrix (IoU)...
        self_dist_matrix = mm.distances.iou_matrix(gt_frame_bboxes, gt_frame_bboxes)
        gt_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                              gt_frame_object_ids, # Predicted objects (IDs)
                              self_dist_matrix)

        if dpm_frame.shape[0] != 0:
          dpm_distance_matrix = mm.distances.iou_matrix(gt_frame_bboxes, dpm_frame_bboxes)
          dpm_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                                 dpm_frame_object_ids, # Predicted objects (IDs)
                                 dpm_distance_matrix)

        if rcnn_frame.shape[0] != 0:
          rcnn_distance_matrix = mm.distances.iou_matrix(gt_frame_bboxes, rcnn_frame_bboxes)
          rcnn_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                                 rcnn_frame_object_ids, # Predicted objects (IDs)
                                 rcnn_distance_matrix)
        if ssd_frame.shape[0] != 0:
          ssd_distance_matrix = mm.distances.iou_matrix(gt_frame_bboxes, ssd_frame_bboxes)
          ssd_accumulator.update(gt_frame_object_ids,   # All ground truth objects (IDs)
                                 ssd_frame_object_ids, # Predicted objects (IDs)
                                 ssd_distance_matrix)



mh = mm.metrics.create()
summary = mh.compute_many([gt_accumulator, dpm_accumulator, rcnn_accumulator, ssd_accumulator],
                          names=['Best possible scores', 'DPM', 'R-CNN', 'MobileNetv2-SSD'],
                          metrics=['mota', 'motp', 'mostly_tracked', 'mostly_lost', 'num_switches', 'num_fragmentations'])

strsummary = mm.io.render_summary(summary,
                                  formatters=mh.formatters,
                                  namemap={'mota':'MOTA', 'motp':'MOTP',
                                           'mostly_tracked':'MT', 'mostly_lost':'ML',
                                           'num_switches': 'ID', 'num_fragmentations':'FM'})

print(strsummary)

with open(args.output_path, 'w') as output_file:
    output_file.write(strsummary)