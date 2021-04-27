import os
from os.path import join as opj
import cv2
import pickle
import sys
sys.path.append('../../../')
from config import cfg
    
def parse_pt(pt_file):
    with open(pt_file,'rb') as f:
        lines = pickle.load(f)
    img_rects = dict()
    for line in lines:
        fid = int(lines[line]['frame'][3:])
        tid = lines[line]['id']
        rect = list(map(lambda x: int(float(x)), lines[line]['bbox']))
        if fid not in img_rects:
            img_rects[fid] = list()
        rect.insert(0, tid)
        img_rects[fid].append(rect)
    return img_rects

def show_res(map_tid):
    show_dict = dict()
    for cid_tid in map_tid:
        iid = map_tid[cid_tid]
        if iid in show_dict:
            show_dict[iid].append(cid_tid)
        else:
            show_dict[iid] = [cid_tid]
    for i in show_dict:
        print('ID{}:{}'.format(i,show_dict[i]))

if __name__ == '__main__':
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    data_dir = cfg.DATA_DIR
    roi_dir = cfg.ROI_DIR

    map_tid = pickle.load(open('test_cluster.pkl', 'rb'))['cluster']
    show_res(map_tid)
    f_w = open(cfg.MCMT_OUTPUT_TXT, 'w')
    cam_paths = os.listdir(data_dir)
    cam_paths = list(filter(lambda x: 'c' in x, cam_paths))
    cam_paths.sort()
    for cam_path in cam_paths:
        cid = int(cam_path.split('.')[0][-3:])
        
        roi = cv2.imread(opj(roi_dir, '{}/roi.jpg'.format(cam_path.split('.')[0][-4:])), 0)
        height, width = roi.shape
        img_rects = parse_pt(opj(data_dir, cam_path,'{}_mot_feat_break.pkl'.format(cam_path)))
        for fid in img_rects:
            tid_rects = img_rects[fid]
            fid = int(fid)+1
            for tid_rect in tid_rects:
                tid = tid_rect[0]
                rect = tid_rect[1:]
                cx = 0.5*rect[0] + 0.5*rect[2]
                cy = 0.5*rect[1] + 0.5*rect[3]
                w = rect[2] - rect[0]
                w = min(w*1.2,w+40)
                h = rect[3] - rect[1]
                h = min(h*1.2,h+40)
                rect[2] -= rect[0]
                rect[3] -= rect[1]
                rect[0] = max(0, rect[0])
                rect[1] = max(0, rect[1])
                x1, y1 = max(0, cx - 0.5*w), max(0, cy - 0.5*h)
                x2, y2 = min(width, cx + 0.5*w), min(height, cy + 0.5*h)
                w , h = x2-x1 , y2-y1

                new_rect = list(map(int, [x1, y1, w, h]))
                # new_rect = rect # 使用原bbox
                rect = list(map(int, rect))
                if (cid, tid) in map_tid:
                    new_tid = map_tid[(cid, tid)]
                    f_w.write(str(cid) + ' ' + str(new_tid) + ' ' + str(fid) + ' ' + ' '.join(map(str, new_rect)) + ' -1 -1' '\n')
    f_w.close()
