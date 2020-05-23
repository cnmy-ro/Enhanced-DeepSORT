import os
import numpy as np
import cv2

import torch
import torchvision

from config import *

###############################################################################

def pre_process(frame, bboxes):
        bboxes = np.array(bboxes)

        transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                     torchvision.transforms.Resize((128,128)),
                                                     torchvision.transforms.ToTensor()])

        crops = []
        for bbox in bboxes:
            x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            try:
                crop = frame[y:y+h, x:x+w, :]
                crop = transforms(crop)


                crops.append(crop)
            except:
                print("except-ing")
                continue

        crops = torch.stack(crops)

        return crops


def generate_detections(encoder, output_dir, bboxes_dir=None):

    sequences = sorted(os.listdir(VEHICLE_DATA_DIR))
    sequences = sequences[:20] # First 20 sequences

    bbox_file_names = sorted(os.listdir(bboxes_dir))

    for s, sequence in enumerate(sequences):
        print("Processing %s -----------------------------------" % sequence)
        sequence_dir = os.path.join(VEHICLE_DATA_DIR, sequence) + '/'

        image_filenames = sorted(os.listdir(sequence_dir))

        detection_file = bboxes_dir + bbox_file_names[s]
        detections_in = np.loadtxt(detection_file, delimiter=',')

        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        #min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()

        for frame_idx in range(1, max_frame_idx+1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx), end='')
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            image = cv2.imread(sequence_dir + image_filenames[frame_idx-1])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            boxes = rows[:, 2:6]
            if boxes.shape[0] < 1:
                continue
            processed_crops = pre_process(image, boxes)
            #print(processed_crops.shape)
            features = encoder.forward_once(processed_crops)
            features = features.detach().cpu().numpy()


            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

            #if frame_idx == 20: break ####
            print("  --  Detections shape", np.array(detections_out).shape)#####
        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(output_filename, np.asarray(detections_out), allow_pickle=True)



###############################################################################

if __name__ == '__main__':

    detector_name = 'RCNN'

    bbox_dir = "./Resources/Vehicles/Bboxes/" + detector_name + "/"
    output_dir = "./Resources/Vehicles/Detections/" + detector_name + "/"

    encoder = torch.load(VEHICLE_ENCODER_PATH, map_location='cpu')
    encoder.eval()

    generate_detections(encoder, output_dir, bboxes_dir=bbox_dir)

