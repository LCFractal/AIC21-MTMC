# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import pickle

import cv2
import numpy as np

from application_util import visualization
import torch
from torchvision.ops import nms
# from fm_tracker.loader import load_txt
from fm_tracker.multitracker import JDETracker
import sys
sys.path.append('../../../')
from config import cfg

def gather_sequence_info(sequence_dir, detection_file, max_frame):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
            int(os.path.splitext(f)[0][3:]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    if max_frame > 0:
        max_frame_idx  = max_frame
    
    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = 2048 # default
    det_feat_dic = pickle.load(open(detection_file, 'rb'))
    bbox_dic = {}
    feat_dic = {}
    for image_name in det_feat_dic:
        frame_index = image_name.split('_')[0]
        frame_index = int(frame_index[3:])
        det_bbox = np.array(det_feat_dic[image_name]['bbox']).astype('float32')
        det_feat = det_feat_dic[image_name]['feat']
        score = det_feat_dic[image_name]['conf']
        score = np.array((score,))
        det_bbox = np.concatenate((det_bbox, score)).astype('float32')
        if frame_index not in bbox_dic:
            bbox_dic[frame_index] = [det_bbox]
            feat_dic[frame_index] = [det_feat]
        else:
            bbox_dic[frame_index].append(det_bbox)
            feat_dic[frame_index].append(det_feat)
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": [bbox_dic, feat_dic],
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms,
        "frame_rate": 10
         # "frame_rate": int(info_dict["frameRate"])
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        display, max_frame_idx, mcmt_cfg):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file, max_frame_idx)
    tracker = JDETracker(min_confidence, seq_info["frame_rate"])
    results = []

    def frame_callback(vis, frame_idx):
        #print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        # detections, feats = load_txt(seq_info["detections"], frame_idx)
        [bbox_dic, feat_dic] = seq_info['detections']
        if frame_idx not in bbox_dic:
            print(f'empty for {frame_idx}')
            return
        detections = bbox_dic[frame_idx]
        feats = feat_dic[frame_idx]

        # Run non-maxima suppression.
        boxes = np.array([d[:4] for d in detections], dtype=float)
        scores = np.array([d[4] for d in detections], dtype=float)
        nms_keep = nms(torch.from_numpy(boxes),
                                torch.from_numpy(scores),
                                iou_threshold=nms_max_overlap).numpy()
        detections = np.array([detections[i] for i in nms_keep], dtype=float)
        feats = np.array([feats[i] for i in nms_keep], dtype=float)
        #print(detections)

        # Update tracker.
        online_targets = tracker.update(detections, feats, frame_idx)
        # Store results.
        for t in online_targets:
            tlwh = t.det_tlwh
            tid = t.track_id
            score = t.score
            #feature = t.smooth_feat
            feature = t.features[-1]
            #vertical = tlwh[2] / tlwh[3] > 1.6
            feature = t.smooth_feat
            if tlwh[2] * tlwh[3] > args.min_box_area:
                results.append([
                    frame_idx, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], score, feature
                ])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    save_pickle(output_file, results, seq_info['sequence_name'], mcmt_cfg)
    f = open(output_file, 'w')
    for row in results:
        feat = row[-1]
        feat_str = ','.join([str(fe) for fe in feat])
        # frame_idx, tid, t, l, w, h, score, feature
        # print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,%s' % (
        #      row[0], row[1], row[2], row[3], row[4], row[5], row[6], feat_str),file=f)
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1' % (
           row[0], row[1], row[2], row[3], row[4], row[5], row[6]),file=f)

def save_pickle(output_file, results, sequence_name, mcmt_cfg):
    """Save pickle."""

    # if not os.path.exists(mot_image_dir):
    #     os.makedirs(mot_image_dir)
    feat_pkl_file = f'{mcmt_cfg.DATA_DIR}/{sequence_name}/{sequence_name}_mot_feat_raw.pkl'
    mot_feat_dic = {}
    for row in results:
        [fid, pid, x, y, w, h] = row[:6]    # pylint: disable=invalid-name
        feat = row[-1]
        image_name = f'{sequence_name}_{pid}_{fid}.png'
        bbox = (x, y, x+w, y+h)
        frame = f'img{int(fid):06d}'
        mot_feat_dic[image_name] = {'bbox': bbox, 'frame': frame, 'id': int(pid),
                                    'imgname': image_name, 'feat': feat}
    pickle.dump(mot_feat_dic, open(feat_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=False)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=False)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--max_frame_idx", help="Maximum size of the frame ids.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument('--min-box-area', type=float, default=50, help='filter out tiny boxes')
    parser.add_argument('--cfg_file', default='aic_mcmt.yml', help='Config file for mcmt')
    parser.add_argument('--seq_name', default='c041', help='Seq name')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(f'../../../config/{args.cfg_file}')
    cfg.freeze()
    args.sequence_dir = os.path.join(cfg.DET_SOURCE_DIR, args.seq_name)
    args.detection_file = os.path.join(cfg.DATA_DIR, args.seq_name,
                                       f'{args.seq_name}_dets_feat.pkl')
    args.output_file = os.path.join(cfg.DATA_DIR, args.seq_name,
                                    f'{args.seq_name}_mot.txt')

    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.display, args.max_frame_idx, cfg)
