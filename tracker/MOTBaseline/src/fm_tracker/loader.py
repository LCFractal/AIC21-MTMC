import os
import numpy as np

def load_seq(path, min_frame_id, max_frame_id):
    seq_det = []
    for frame in range(min_frame_id, max_frame_id+1):
        seq_det.append(load_txt(path, frame))
    return seq_det

def load_txt(path, frame):
    path = os.path.join(path, f'{frame-1:05}.txt')
    frame_det, frame_feat = [], []
    f = open(path)
    for line in f.readlines():
        line = line.strip().split()
        box = np.fromstring(line[0], dtype=np.float32, sep=',')
        feat = np.fromstring(line[1], dtype=np.float32, sep=',')
        frame_det.append(box)
        frame_feat.append(feat)
    f.close()
    return frame_det, frame_feat


if __name__ == '__main__':
    load_seq('/home/jiasheng.tjs/data/fairmot_outs/MOT17-02-SDP/', 1, 500)
