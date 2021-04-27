import os
from os.path import join as opj
import numpy as np
import pickle
from utils.zone_intra import zone
import sys
sys.path.append('../../../')
from config import cfg

def parse_pt(pt_file,zones):
    if not os.path.isfile(pt_file):
        return dict()
    with open(pt_file,'rb') as f:
        lines = pickle.load(f)
    mot_list = dict()
    for line in lines:
        fid = int(lines[line]['frame'][3:])
        tid = lines[line]['id']
        bbox = list(map(lambda x:int(float(x)), lines[line]['bbox']))
        if tid not in mot_list:
            mot_list[tid] = dict()
        out_dict = lines[line]
        out_dict['zone'] = zones.get_zone(bbox)
        mot_list[tid][fid] = out_dict
    return mot_list

def parse_bias(timestamp_dir, scene_name):
    cid_bias = dict()
    for sname in scene_name:
        with open(opj(timestamp_dir, sname + '.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                cid = int(line[0][2:])
                bias = float(line[1])
                if cid not in cid_bias: cid_bias[cid] = bias
    return cid_bias

def out_new_mot(mot_list,mot_path):
    out_dict = dict()
    for tracklet in mot_list:
        tracklet = mot_list[tracklet]
        for f in tracklet:
            out_dict[tracklet[f]['imgname']]=tracklet[f]
    pickle.dump(out_dict,open(mot_path,'wb'))

if __name__ == '__main__':
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    scene_name = ['S06']
    data_dir = cfg.DATA_DIR
    save_dir = './exp/viz/test/S06/trajectory/'
    cid_bias = parse_bias(cfg.CID_BIAS_DIR, scene_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    cam_paths = os.listdir(data_dir)
    cam_paths = list(filter(lambda x: 'c' in x, cam_paths))
    cam_paths.sort()
    zones = zone()

    for cam_path in cam_paths:
        print('processing {}...'.format(cam_path))
        cid = int(cam_path[-3:])
        f_w = open(opj(save_dir, '{}.pkl'.format(cam_path)), 'wb')
        cur_bias = cid_bias[cid]
        mot_path = opj(data_dir, cam_path,'{}_mot_feat.pkl'.format(cam_path))
        new_mot_path = opj(data_dir, cam_path, '{}_mot_feat_break.pkl'.format(cam_path))
        print(new_mot_path)
        zones.set_cam(cid)
        mot_list = parse_pt(mot_path,zones)
        mot_list = zones.break_mot(mot_list, cid)
        # mot_list = zones.comb_mot(mot_list, cid)
        mot_list = zones.filter_mot(mot_list, cid) # filter by zone
        mot_list = zones.filter_bbox(mot_list, cid)  # filter bbox
        out_new_mot(mot_list, new_mot_path)

        tid_data = dict()
        for tid in mot_list:
            if cid not in [41,43,46,42,44,45]:
                break
            tracklet = mot_list[tid]
            if len(tracklet) <= 1: continue

            frame_list = list(tracklet.keys())
            frame_list.sort()
            # if tid==11 and cid==44:
            #     print(tid)
            zone_list = [tracklet[f]['zone'] for f in frame_list]
            feature_list = [tracklet[f]['feat'] for f in frame_list if (tracklet[f]['bbox'][3]-tracklet[f]['bbox'][1])*(tracklet[f]['bbox'][2]-tracklet[f]['bbox'][0])>2000]
            if len(feature_list)<2:
                feature_list = [tracklet[f]['feat'] for f in frame_list]
            io_time = [cur_bias + frame_list[0] / 10., cur_bias + frame_list[-1] / 10.]
            all_feat = np.array([feat for feat in feature_list])
            mean_feat = np.mean(all_feat, axis=0)

            tid_data[tid]={
                'cam': cid,
                'tid': tid,
                'mean_feat': mean_feat,
                'zone_list':zone_list,
                'frame_list': frame_list,
                'tracklet': tracklet,
                'io_time': io_time
            }

        pickle.dump(tid_data,f_w)
        f_w.close()
