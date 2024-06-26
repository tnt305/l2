import os
import random
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from ensemble_boxes import weighted_boxes_fusion
from utils import load_dict

AICITY_CLASSES = ['motorbike', 'DHelmet', 'DNoHelmet', 'P1Helmet', 'P1NoHelmet', 'P2Helmet', 'P2NoHelmet', 'P0Helmet', 'P0NoHelmet']
HEAD_CLASSES = ['head', 'helmet', 'uncertain']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(8)]

def run_wbf(image_ids, box_pred, score_pred, label_pred, NMS_THRESH, BOX_THRESH, PP_THRESH):
    output_dict = {}
    for image_id in image_ids:
        if image_id not in box_pred.keys(): continue

        boxes = box_pred[image_id]
        scores = score_pred[image_id]
        labels = label_pred[image_id]
        list_box = []
        list_score = []
        list_label = []
        for box in boxes:
            for b in box:
                list_box.append(b)
        for score in scores:
            for s in score:
                list_score.append(s)
        for label in labels:
            for l in label:
                list_label.append(l)

        boxes = np.array(list_box)
        scores = np.array(list_score)
        labels = np.array(list_label)
        remove_idx = []
        for idx, label in enumerate(list_label):
            if int(label) == 5: continue
            remove_idx.append(idx)
        boxes = np.delete(boxes, remove_idx, axis=0)
        scores = np.delete(scores, remove_idx, axis=0)
        labels = np.delete(labels, remove_idx, axis=0)
        boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=NMS_THRESH,
                                                      skip_box_thr=BOX_THRESH)

        idxs = np.where(scores > PP_THRESH)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]

        if len(boxes) > 0:
            height, width = 1080, 1920
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * width).clip(min=0, max=width - 1)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * height).clip(min=0, max=height - 1)
        output_dict[image_id] = (boxes, scores, labels)

    return output_dict

def emsemble_obj(output_txt_file, image_ids):

    effdet_ed6_640_box_pred = load_dict('pkl/effdet_ed6_640_box_pred.pkl')
    effdet_ed6_640_score_pred = load_dict('pkl/effdet_ed6_640_score_pred.pkl')
    effdet_ed6_640_label_pred = load_dict('pkl/effdet_ed6_640_label_pred.pkl')

    output_dict = run_wbf(image_ids, effdet_ed6_640_box_pred, effdet_ed6_640_score_pred,
                          effdet_ed6_640_label_pred, NMS_THRESH=0.5, BOX_THRESH=0.1, PP_THRESH=0.1)

    for id, obj in output_dict.items():
        bboxs, scores, labels = obj[0], obj[1], obj[2]
        for idx, bbox in enumerate(bboxs):
            video_id = id.split("_")[0]
            frame_id = id.split("_")[1]
            content_line = '{},{},{},{},{},{},{},{}'.format(int(video_id), int(frame_id), int(bbox[0]),
                                                            int(bbox[1]), int(bbox[2]) - int(bbox[0]),
                                                            int(bbox[3]) - int(bbox[1]), int(labels[idx]),
                                                            scores[idx])
            output_txt_file.write(content_line)
            output_txt_file.write('\n')


if __name__ == '__main__':
    df = '../aicity_dataset/aicity2023_track5_test_images/'
    image_file_list = [f for f in listdir(df) if isfile(join(df, f))]
    image_ids = []
    for id in image_file_list:
        image_ids.append(id.split(".")[0])
    txt_path = 'effdet_ed6_640_pseudo.txt'
    output_txt_file = open(txt_path, "w")
    emsemble_obj(output_txt_file, image_ids)



