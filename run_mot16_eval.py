import numpy as np

import motmetrics as mm

'''
Specs:
    2 accumulators - one for default detection results, one for SSD
'''

acc = mm.MOTAccumulator(auto_id=False)


acc.update([1,2],   # All ground truth objects (IDs)
           [1,2,3], # Predicted objects (IDs)
           [ [0.1, np.nan, 0.3],
             [0.5, 0.2, 0.3] ],
           frameid=1)


print(acc.mot_events)