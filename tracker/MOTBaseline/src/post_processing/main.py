
import numpy as np
from track_nms import track_nms
from post_association import associate
import os
from tqdm import tqdm
import pandas as pd
from interploation import *
import pickle
import sys
sys.path.append('../../../../')
from config import cfg

def load_raw_mot(seq, mcmt_cfg):
    """Load pickle to speed up."""

    mot_feat_dic = pickle.load(open(f'{mcmt_cfg.DATA_DIR}/{seq}/{seq}_mot_feat_raw.pkl', 'rb'))
    results = []
    for image_name in sorted(list(mot_feat_dic.keys())):
        feat_dic = mot_feat_dic[image_name]
        frame = int(feat_dic['frame'][3:])
        pid = int(feat_dic['id'])
        bbox = np.array(feat_dic['bbox']).astype('float32')
        feat = feat_dic['feat']
        dummpy_input = np.array([frame, pid, bbox[0], bbox[1], bbox[2]-bbox[0],
                                bbox[3]-bbox[1], -1, -1, -1, -1])
        dummpy_input = np.concatenate((dummpy_input, feat))
        results.append(dummpy_input)
    return np.array(results)

def eval_seq(seq, pp='', split='train', mcmt_cfg=None):
    if pp == 'pp':
        use_pp = True
        print('using post processing')
    else:
        use_pp = False
        print('NO post processing')
    trk_file = f'{mcmt_cfg.DATA_DIR}/{seq}/{seq}_mot.txt'
    print('loading tracked file ' + trk_file)
    # results = np.loadtxt(trk_file, delimiter=',')
    results = load_raw_mot(seq, mcmt_cfg)
    if not use_pp:
        # remove len 1 track and interpolate
        # results = interpolate_traj(results)
        # Store results.
        trk_dir = os.path.dirname(trk_file)
        if not os.path.exists(os.path.join(trk_dir, 'res')):
            os.makedirs(os.path.join(trk_dir, 'res'))
        output_file = os.path.join(trk_dir, 'res', os.path.basename(trk_file))
        f = open(output_file, 'w')
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
        f.close()
        return
    loaded_trk_ids = np.unique(results[:, 1])

    mark_interpolation=False
    # post associate
    results = associate(results, 0.1, 10, seq)
    results = associate(results, 0.1, 10, seq)

    # remove len 1 track and interpolate, help on reducing FNs.
    #results = interpolate_traj(results, drop_len=1)

    # track nms can help reduce FPs.
    results = track_nms(results, 0.65)

    trk_ids = np.unique(results[:, 1])
    print('after all PP, merging ', len(loaded_trk_ids) - len(trk_ids), ' tracks')
    
    # Store results.
    save_pickle(results, seq, mcmt_cfg)
    trk_dir = os.path.dirname(trk_file)
    if not os.path.exists(os.path.join(trk_dir, 'res')):
        os.makedirs(os.path.join(trk_dir, 'res'))
    output_file = os.path.join(trk_dir, 'res', os.path.basename(trk_file))
    f = open(output_file, 'w')
    for row in results:
        if mark_interpolation:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,%d,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6]),file=f)
        else:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    f.close()

def save_pickle(results, sequence_name, mcmt_cfg):
    """Save pickle."""

    feat_pkl_file = f'{mcmt_cfg.DATA_DIR}/{sequence_name}/{sequence_name}_mot_feat.pkl'
    mot_feat_dic = {}
    for row in results:
        [fid, pid, x, y, w, h] = row[:6]    # pylint: disable=invalid-name
        fid = int(fid)
        pid = int(pid)
        feat = np.array(row[-2048:])
        image_name = f'{sequence_name}_{pid}_{fid}.png'
        bbox = (x, y, x+w, y+h)
        frame = f'img{int(fid):06d}'
        mot_feat_dic[image_name] = {'bbox': bbox, 'frame': frame, 'id': pid,
                                    'imgname': image_name, 'feat': feat}
    pickle.dump(mot_feat_dic, open(feat_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    import sys
    print('sys args ars: ', sys.argv)
    cfg.merge_from_file(f'../../../../config/{sys.argv[3]}')
    cfg.freeze()
    eval_seq(sys.argv[1], pp=sys.argv[2], mcmt_cfg=cfg)
