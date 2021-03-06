#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
from collections import defaultdict
from fvcore.common.file_io import PathManager

import numpy as np

logger = logging.getLogger(__name__)

FPS = 30
AVA_VALID_FRAMES = range(902, 1799)


def load_image_lists(cfg, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    list_filenames = [
        os.path.join(cfg.AVA.FRAME_LIST_DIR, filename)
        for filename in (
            cfg.AVA.TRAIN_LISTS if is_train else cfg.AVA.TEST_LISTS
        )
    ]
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    labels = defaultdict(list)
    for list_filename in list_filenames:
        with PathManager.open(list_filename, "r") as f:
            f.readline()
            for line in f:
                row = line.split('\n')[0].split(',')
                # The format of each row should follow:
                # original_vido_id video_id frame_id path labels.
                assert len(row) == 3
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                data_key = video_name_to_idx[video_name]

                image_paths[data_key].append(
                    os.path.join(cfg.AVA.FRAME_DIR, video_name + '_' + row[1]+'.jpg')
                )
                labels[data_key].append(int(row[2]) - 1)

    image_paths = [image_paths[i] for i in range(len(image_paths))]
    logger.info(
        "Finished loading image paths from: %s" % ", ".join(list_filenames)
    )

    return image_paths, video_idx_to_name, labels


def load_boxes_and_labels(cfg, mode):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    gt_lists = cfg.AVA.TRAIN_GT_BOX_LISTS if mode == "train" else []
    pred_lists = (
        cfg.AVA.TRAIN_PREDICT_BOX_LISTS
        if mode == "train"
        else cfg.AVA.TEST_PREDICT_BOX_LISTS
    )
    ann_filenames = [
        os.path.join(cfg.AVA.ANNOTATION_DIR, filename)
        for filename in gt_lists + pred_lists
    ]
    ann_is_gt_box = [True] * len(gt_lists) + [False] * len(pred_lists)

    detect_thresh = cfg.AVA.DETECTION_SCORE_THRESH
    all_boxes = {}
    count = 0
    unique_box_count = 0
    for filename, is_gt_box in zip(ann_filenames, ann_is_gt_box):
        with PathManager.open(filename, "r") as f:
            for line in f:
                row = line.strip().split(",")
                # When we use predicted boxes to train/eval, we need to
                # ignore the boxes whose scores are below the threshold.
                if not is_gt_box:
                    score = float(row[7])
                    if score < detect_thresh:
                        continue

                video_name, frame_sec = row[0], int(row[1])

                # Only select frame_sec % 4 = 0 samples for validation if not
                # set FULL_TEST_ON_VAL.
                if (
                    mode == "val"
                    and not cfg.AVA.FULL_TEST_ON_VAL
                    and frame_sec % 4 != 0
                ):
                    continue

                # Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
                box_key = ",".join(row[2:6])
                box = list(map(float, row[2:6]))
                label = -1 if row[6] == "" else int(row[6])

                if video_name not in all_boxes:
                    all_boxes[video_name] = {}
                    for sec in AVA_VALID_FRAMES:
                        all_boxes[video_name][sec] = {}

                if box_key not in all_boxes[video_name][frame_sec]:
                    all_boxes[video_name][frame_sec][box_key] = [box, []]
                    unique_box_count += 1

                all_boxes[video_name][frame_sec][box_key][1].append(label)
                if label != -1:
                    count += 1

    for video_name in all_boxes.keys():
        for frame_sec in all_boxes[video_name].keys():
            # Save in format of a list of [box_i, box_i_labels].
            all_boxes[video_name][frame_sec] = list(
                all_boxes[video_name][frame_sec].values()
            )

    logger.info(
        "Finished loading annotations from: %s" % ", ".join(ann_filenames)
    )
    logger.info("Detection threshold: {}".format(detect_thresh))
    logger.info("Number of unique boxes: %d" % unique_box_count)
    logger.info("Number of annotations: %d" % count)

    return all_boxes


def get_keyframe_data(boxes_and_labels, type_labels, seq_len, num_frames, output_size, predict_all):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        keyframe_boxes_and_labels.append([])
        current_label = boxes_and_labels[video_idx][0]
        ix_st = 0
        if not predict_all:
            for sec, label in enumerate(boxes_and_labels[video_idx]):
                    if ix_st == -1:
                        ix_st = sec
                        current_label = label

                    if label != current_label:
                        if type_labels == 'class':
                            keyframe_indices.append(
                                (video_idx, current_label, ix_st, sec)
                            )
                            keyframe_boxes_and_labels[video_idx].append(
                                label
                            )
                        elif type_labels == 'mask':
                            left_ix, right_ix = move_window_given(sec, ix_st, len(boxes_and_labels[video_idx]),
                                                                seq_len, video_idx)

                            labels = np.array(boxes_and_labels[video_idx][left_ix: right_ix])
                            keyframe_indices.append(
                                (video_idx, labels, left_ix, right_ix)
                            )
                            keyframe_boxes_and_labels[video_idx].append(
                                labels
                            )
                        elif type_labels == 'regression':
                            left_ix, right_ix = move_window_given(sec, ix_st, len(boxes_and_labels[video_idx]),
                                                                seq_len, video_idx)
                            labels = np.array(boxes_and_labels[video_idx][left_ix: right_ix])
                            lbs_ixs = np.where(np.diff(labels) != 0)[0] + 1
                            if len(lbs_ixs) == 0:
                                lbs_regress = np.array([1, num_frames, *[0]*(output_size-1), labels[0], *[0]*(output_size-1)])
                            else:
                                num_add = output_size - len(lbs_ixs) - 1
                                if num_add < 0:
                                    raise ValueError(f'No more then {output_size} actions should be added in one element.')
                                lbs_val = np.array([labels[0], *labels[lbs_ixs], *[0]*num_add])
                                lbs_length = np.array([lbs_ixs[0], *np.diff(np.array([*lbs_ixs, num_frames])), *[0]*num_add])
                                lbs_regress = np.array([output_size-num_add, *lbs_length, *lbs_val])
                            keyframe_indices.append(
                                (video_idx, lbs_regress, left_ix, right_ix)
                            )
                            keyframe_boxes_and_labels[video_idx].append(
                                lbs_regress
                            )
                        elif type_labels == 'length':
                            left_ix, right_ix = move_window_given(sec, ix_st, len(boxes_and_labels[video_idx]),
                                                                seq_len, video_idx)
                            labels = np.array(boxes_and_labels[video_idx][left_ix: right_ix])
                            lbs_ixs = np.where(np.diff(labels) != 0)[0] + 1
                            if len(lbs_ixs) == 0:
                                lbs_length = np.array([1, num_frames, *[0]*(output_size-1)])
                            else:
                                num_add = output_size - len(lbs_ixs) - 1
                                if num_add < 0:
                                    raise ValueError(f'No more then {output_size} actions should be added in one element.')
                                lbs_length = np.array([output_size-num_add, lbs_ixs[0],
                                                    *np.diff(np.array([*lbs_ixs, num_frames])), *[0]*num_add])
                            keyframe_indices.append(
                                (video_idx, lbs_length, left_ix, right_ix)
                            )
                            keyframe_boxes_and_labels[video_idx].append(
                                lbs_length
                            )
                        elif type_labels in ['stend', 'pipeline']:
                            left_ix, right_ix = move_window_given(sec, ix_st, len(boxes_and_labels[video_idx]),
                                                                  seq_len, video_idx)
                            labels = np.array(boxes_and_labels[video_idx][left_ix: right_ix])
                            ix_stend = np.where(np.diff(labels) != 0)[0]

                            start_ix = np.zeros_like(labels, dtype=np.float32)
                            end_ix = np.zeros_like(labels, dtype=np.float32)

                            start_ix[[*np.clip(ix_stend+1, 0, len(labels)-1),
                                      *np.clip(ix_stend+2, 0, len(labels)-1),
                                      *np.clip(ix_stend+3, 0, len(labels)-1)]] = 1
                            end_ix[[*np.clip(ix_stend, 0, len(labels)-1),
                                    *np.clip(ix_stend-1, 0, len(labels)-1),
                                    *np.clip(ix_stend-2, 0, len(labels)-1)]] = 1


                            keyframe_indices.append(
                                (video_idx, [labels, start_ix, end_ix], left_ix, right_ix)
                            )
                            keyframe_boxes_and_labels[video_idx].append(
                                [start_ix, end_ix]
                            )
                        ix_st = -1
        else:
            iterator = np.arange(0, len(boxes_and_labels[video_idx]),  num_frames)
            rest = len(boxes_and_labels[video_idx]) % num_frames

            cropped_labels = boxes_and_labels[video_idx][:-rest] if rest != 0 else boxes_and_labels[video_idx]
            cropped_labels = np.array(cropped_labels).reshape(-1, num_frames)
            for ix, iterat in enumerate(iterator[:-1]):
                keyframe_indices.append(
                    (video_idx, cropped_labels[ix], iterat, iterator[ix+1])
                )
                keyframe_boxes_and_labels[video_idx].append(
                                cropped_labels[ix]
                            )

    return keyframe_indices, keyframe_boxes_and_labels

def move_window_given(sec, ix_st, len_b_and_l, seq_len, idx):
    if seq_len <= (sec - ix_st):
        left_ix = ix_st
        right_ix = ix_st + seq_len
    else:
        max_off = seq_len - (sec - ix_st)
        offset = np.random.randint(0, max_off)
        if offset > ix_st:
            left_ix = 0
            right_ix = seq_len
        elif sec + (max_off - offset) > len_b_and_l:
            right_ix = len_b_and_l
            left_ix = right_ix - seq_len
            if left_ix < 0:
                raise ValueError(f'Given video is too short. Video idx: {idx}')
        else:
            left_ix = ix_st - offset
            right_ix = sec + (max_off - offset)
    return left_ix, right_ix

def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count
