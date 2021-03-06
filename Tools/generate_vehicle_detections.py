import os
import numpy as np
import cv2

import torch
import torchvision

from config import VEHICLE_DATA_DIR, VEHICLE_ENCODER_PATH

###############################################################################

def pre_process(frame, bboxes, transforms):
        bboxes = np.array(bboxes)
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
    sequence_names = sorted(os.listdir(VEHICLE_DATA_DIR))

    sequence_names = sequence_names[19:20] # First 20 sequences

    bbox_file_names = sorted(os.listdir(bboxes_dir))

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                     torchvision.transforms.Resize((128,128)),
                                                     torchvision.transforms.ToTensor()])
    encoder.eval()

    for sequence in sequence_names:
        print("Processing %s -----------------------------------" % sequence)
        sequence_dir = os.path.join(VEHICLE_DATA_DIR, sequence) + '/'

        image_filenames = sorted(os.listdir(sequence_dir))

        detection_file = bboxes_dir + sequence + ".txt"
        # detection_file = bboxes_dir + bbox_file_names[start_from_frame_idx + s]
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        #min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()        skipped_frames = 0

        for frame_idx in range(1, max_frame_idx+1):
            print("\nFrame %05d/%05d" % (frame_idx, max_frame_idx), end='')
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            image = cv2.imread(sequence_dir + image_filenames[frame_idx-1])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes = rows[:, 2:6]
            if rows.shape[0] <= 1: # If there' just one or no detected object -- results is irregularly shaped detection_out array
                print(" - skipping frame", end='')
                skipped_frames += 1
                continue            # So skip this frame

            processed_crops = pre_process(image, boxes, transforms)
            features = encoder.forward_once(processed_crops)
            features = features.detach().cpu().numpy()


            detections_out += [np.r_[(row, feature)] for row, feature
                                in zip(rows, features)]

            print("  --  Detections shape", np.array(detections_out).shape)#####
        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(output_filename, np.asarray(detections_out), allow_pickle=True)
        print("Skipped frames:", skipped_frames)



###############################################################################

if __name__ == '__main__':

    detector_name = 'SSD'

    bbox_dir = "./Resources/Vehicles/Bboxes/" + detector_name + "/"
    output_dir = "./Resources/Vehicles/Detections/" + detector_name + "/"

    encoder = torch.load(VEHICLE_ENCODER_PATH, map_location='cpu')

    generate_detections(encoder, output_dir, bboxes_dir=bbox_dir)

