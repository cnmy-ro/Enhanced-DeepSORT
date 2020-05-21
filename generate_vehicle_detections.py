import os
import numpy as np
import cv2

import torch
import torchvision

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


def generate_detections(encoder, data_dir, output_dir, detection_dir=None):

    for sequence in sorted(os.listdir(data_dir))[13:]:
        print("Processing %s -----------------------------------" % sequence)
        sequence_dir = os.path.join(data_dir, sequence) + '/'

        image_filenames = sorted(os.listdir(sequence_dir))
        #print("Image names:", image_filenames)

        detection_file = detection_dir + sequence + "_Det_DPM.txt"
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_in = detections_in[detections_in[:,-1] > 0.1] # 10% confidence threshold

        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        #min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()

        for frame_idx in range(1, max_frame_idx):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
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

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(output_filename, np.asarray(detections_out))


###############################################################################

## DPM bboxes
dpm_bboxes = np.loadtxt("UA-DETRAC/Object Data/Bboxes/DPM/MVI_20011_Det_DPM.txt", delimiter=',')
dpm_bboxes = dpm_bboxes[dpm_bboxes[:,0] == 1]
dpm_bboxes = dpm_bboxes[dpm_bboxes[:,-1] > 0.1] # 10% confidence threshold


###############################################################################

if __name__ == '__main__':

    detector_name = 'DPM'

    data_dir = "UA-DETRAC/Insight-MVT_Annotation_Train/"
    bbox_dir = "UA-DETRAC/Object Data/Bboxes/" + detector_name + "/"
    output_dir = "UA-DETRAC/Object Data/Detections/" + detector_name + "/"


    encoder = torch.load('./vehicle_encoder_model/ckpts/model640.pt', map_location='cpu')
    encoder.eval()

    generate_detections(encoder, data_dir, output_dir, detection_dir=bbox_dir)

