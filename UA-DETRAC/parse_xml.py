import os
import numpy as np
import xml.etree.ElementTree as ET


def parse_labels(label_file):
    """
    Returns a set of metadata (1 per track) and a list of labels (1 item per
    frame, where an item is a list of dictionaries (one dictionary per object
    with fields id, class, truncation, orientation, and bbox
    """
    tree = ET.parse(label_file)
    root = tree.getroot()

    # get sequence attributes
    seq_name = root.attrib['name']

    # get list of all frame elements
    frames = root.getchildren()

    # first child is sequence attributes
    seq_attrs = frames[0].attrib

    # second child is ignored regions
    ignored_regions = []
    for region in frames[1]:
        coords = region.attrib
        box = np.array([float(coords['left']),
                        float(coords['top']),
                        float(coords['left']) + float(coords['width']),
                        float(coords['top'])  + float(coords['height'])])
        ignored_regions.append(box)
    frames = frames[2:]

    # rest are bboxes
    all_boxes = []
    for frame in frames:
        frame_boxes = []
        boxids = frame.getchildren()[0].getchildren()
        for boxid in boxids:
            data = boxid.getchildren()
            coords = data[0].attrib
            stats = data[1].attrib
            bbox = np.array([float(coords['left']),
                            float(coords['top']),
                            float(coords['width']),
                            float(coords['height'])])
            '''
            det_dict = {
                    'id':int(boxid.attrib['id']),
                    'class':stats['vehicle_type'],
                    'color':stats['color'],
                    'orientation':float(stats['orientation']),
                    'truncation':float(stats['truncation_ratio']),
                    'bbox':bbox
                    }
            '''
            det_dict = {
                    'id':int(boxid.attrib['id']),
                    'bbox':bbox
                    }

            frame_boxes.append(det_dict)
        all_boxes.append(frame_boxes)

    sequence_metadata = {
            'sequence':seq_name,
            'seq_attributes':seq_attrs,
            'ignored_regions':ignored_regions
            }
    return all_boxes, sequence_metadata



#################################################################

gt_dir = "DETRAC-Train-Annotations-XML/"
file_names = os.listdir(gt_dir)


for file_name in file_names: # For each video
    print("Processing ground-truth file: ", file_name)

    all_boxes, sequence_metadata = parse_labels(gt_dir+file_name)
    out_file_path = "Object Data/ground_truths/" + file_name.split('.')[0] + ".txt"
    with open(out_file_path, 'w') as out_f:
        for i, det_dict_list in enumerate(all_boxes): # FOr each frame in that video
            frame = i+1
            for det_dict in det_dict_list: # For each object in that frame
                object_id = det_dict['id']
                x,y,w,h = det_dict['bbox']
                out_f.write("{},{},{},{},{},{}\n".format(frame, object_id, x,y,w,h))
