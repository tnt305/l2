import numpy as np
import pickle
# from evaluation import calculate_final_score
from ensemble_boxes import weighted_boxes_fusion
import os
import pandas as pd
from PIL import Image
import pickle
import torch


class MyThresh:
    def __init__(self, index, total, nms_thresh, box_thresh, pp_threshold):
        self.index = index
        self.total = total
        self.nms_thresh = nms_thresh
        self.box_thresh = box_thresh
        self.pp_threshold = pp_threshold


def wbf_optimize(mt, ground_truth, box_pred, score_pred, label_pred):
    all_predictions = []
    for image_id in np.unique(list(ground_truth.keys())):
        gt = ground_truth[image_id]
        boxes = box_pred[image_id]
        scores = score_pred[image_id]
        labels = label_pred[image_id]

        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=mt.nms_thresh,
                                                      skip_box_thr=mt.box_thresh)

        # if len(boxes) > 0:
        boxes[:, 0] = boxes[:, 0] * 1920
        boxes[:, 1] = boxes[:, 1] * 1080
        boxes[:, 2] = boxes[:, 2] * 1920
        boxes[:, 3] = boxes[:, 3] * 1080

        all_predictions.append({
            'pred_boxes': boxes.astype(int),
            'scores': scores,
            'gt_boxes': gt.astype(int),
            'labels': labels
        })

    final_score = calculate_final_score(all_predictions, score_threshold=mt.pp_threshold)
    # print('{:05d}/{:05d} :  | NMS {:.2f} | CONF {:.2f} | PP {:.2f}'.format(mt.index, mt.total, mt.nms_thresh, mt.box_thresh, mt.pp_threshold))
    # print(final_score)
    return final_score


def save_dict(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.5f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)


def collate_fn(batch):
    return tuple(zip(*batch))


def save_dict(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def get_resolution(image_id, test_dir):
    img_path = '{}/{}.jpg'.format(test_dir, image_id)
    image = Image.open(img_path)
    image = image.convert('RGB')
    image = np.array(image)
    height, width = image.shape[0:2]
    del image
    return image_id, height, width


def refine_checkpoint_in(ckpt_input, ckpt_output):
    ckpt = torch.load(ckpt_input)
    torch.save({
        'model': ckpt,
        'optimizer': None,
        'val_loss_min': None,
    }, ckpt_output)
    del ckpt


def refine_checkpoint_out(ckpt_input, ckpt_output):
    ckpt = torch.load(ckpt_input)
    torch.save(ckpt['model'], ckpt_output)
    del ckpt


def make_pseudo_dataframe(image_ids, output_dict, TEST_DIR, df, TRAIN_DIR, PSEUDO_FOLD):
    results = []
    for image_id in image_ids:
        if image_id not in output_dict.keys(): continue
        boxes, scores, labels = output_dict[image_id]
        if boxes.shape[0] == 0:
            continue
        else:
            for idx, box in enumerate(boxes):
                result = {
                    'image_path': os.path.join(TEST_DIR, image_id + '.jpg'),
                    'xmin': float(int(box[0])),
                    'ymin': float(int(box[1])),
                    'xmax': float(int(box[2])),
                    'ymax': float(int(box[3])),
                    'isbox': True,
                    'label': labels[idx]
                }
                results.append(result)
    pseudo_df = pd.DataFrame(results, columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'isbox', 'label'])

    img_paths = []
    for image_id in df.image_id.values:
        img_paths.append(os.path.join(TRAIN_DIR, image_id + '.jpg'))
    df['image_path'] = np.array(img_paths)
    valid_df = df.loc[df['fold'] == PSEUDO_FOLD]
    train_df = df.loc[~df.index.isin(valid_df.index)]
    valid_df = valid_df.loc[valid_df['isbox'] == True]

    train_df = train_df[['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'isbox', 'label']].reset_index(drop=True)
    valid_df = valid_df[['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'isbox', 'label']].reset_index(drop=True)

    train_df = pd.concat([train_df, pseudo_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
    train_df.to_csv('train.csv', index=False)
    valid_df.to_csv('valid.csv', index=False)


    def create_dataset(gt_filepath, head=False):
        with open(gt_filepath, 'r') as text_file:
            lines = text_file.readlines()

        data = [line.strip().split(',') for line in lines]

        with open('gt.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(data)

        df = pd.read_csv('gt.csv', names=['video_id', 'frame', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'class'])

        train_dataset = pd.DataFrame(columns=['image_id', 'folds', 'x_min', 'y_min', 'x_max', 'y_max', 'isbox', 'source', 'label'])

        train_dataset['image_id'] = df['video_id'].astype(str) + '_' + df['frame'].astype(str)
        train_dataset['folds'] = df['video_id'] % 5
        train_dataset['x_min'] = df['bb_left'].astype(float)
        train_dataset['y_min'] = df['bb_top'].astype(float)
        train_dataset['x_max'] = (df['bb_left'] + df['bb_width']).astype(float)
        train_dataset['y_max'] = (df['bb_top'] + df['bb_height']).astype(float)
        train_dataset['isbox'] = True
        if head == False:
            train_dataset['source'] = df['class']
        else:
            train_dataset['source'] = 0

        train_dataset['label'] = df['class']
        train_dataset = train_dataset.drop(train_dataset.columns[:7], axis=1)

        filename = 'trainset_ai.csv' if head == False else 'trainset_head.csv'
        train_dataset.to_csv(filename, index=False)