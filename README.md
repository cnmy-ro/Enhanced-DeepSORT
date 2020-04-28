# Enhanced Deep-SORT for Vehicle Tracking

## Objectives
1. Create an object detection pipeline (based on MobileNet-SSD) and integrate it with Deep-SORT. Provide two modes of operation: Eval mode for benchmarking; Cam mode for deployment. Evaluate quantitatively on MOT16, compare with original.
2. Adapt Deep-SORT to track vehicles -  requires changing the CNN encoder model. Try to integrate an existing vehicle-specific model into the code. Then build and train my own model and use that instead. Evaluate qualitatively.
3. Implement Confidence Trigger Detection mechanism for speed-up. Measure the improvement.

## New tools to use
1. Logging module for Python
2. Tensorflow 2.0
3. RWTH Cluster

## Useful links

#### Main Data set
- MOT16 Multiple Object (Pedestrian) Tracking benchmark: [page](https://motchallenge.net/data/MOT16/) | [3rd party evaluation code](https://github.com/cheind/py-motmetrics)

#### Trackers
- DeepSORT pedestrian tracking: [paper](https://arxiv.org/abs/1703.07402) | [code](https://github.com/nwojke/deep_sort)
- SORT multi-object tracking: [paper](https://arxiv.org/abs/1602.00763) | [code](https://github.com/abewley/sort)

#### Object detector
- MobileNet-SSD [TF object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) | [Model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

#### Enhancement for real-time performance
- Confidence Trigger Detection [paper](https://arxiv.org/abs/1902.00615)


#### For vehicle tracking
- Deep-SORT for vehicle tracking [code](https://github.com/abhyantrika/nanonets_object_tracking)
