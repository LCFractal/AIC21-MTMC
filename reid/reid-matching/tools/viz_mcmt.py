import os
import sys
sys.path.append('../../../')
from config import cfg
import cv2
import argparse

COLORS_10 = [(144, 238, 144), (178,  34,  34), (221, 160, 221), (  0, 255,   0), (  0, 128,   0), (210, 105,  30), (220,  20,  60),
             (192, 192, 192), (255, 228, 196), ( 50, 205,  50), (139,   0, 139), (100, 149, 237), (138,  43, 226), (238, 130, 238),
             (255,   0, 255), (  0, 100,   0), (127, 255,   0), (255,   0, 255), (  0,   0, 205), (255, 140,   0), (255, 239, 213),
             (199,  21, 133), (124, 252,   0), (147, 112, 219), (106,  90, 205), (176, 196, 222), ( 65, 105, 225), (173, 255,  47),
             (255,  20, 147), (219, 112, 147), (186,  85, 211), (199,  21, 133), (148,   0, 211), (255,  99,  71), (144, 238, 144),
             (255, 255,   0), (230, 230, 250), (  0,   0, 255), (128, 128,   0), (189, 183, 107), (255, 255, 224), (128, 128, 128),
             (105, 105, 105), ( 64, 224, 208), (205, 133,  63), (  0, 128, 128), ( 72, 209, 204), (139,  69,  19), (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (  0, 255, 255), (135, 206, 235), (  0, 191, 255), (176, 224, 230), (  0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (  0, 139, 139), (143, 188, 143), (255,   0,   0), (240, 128, 128),
             (102, 205, 170), ( 60, 179, 113), ( 46, 139,  87), (165,  42,  42), (178,  34,  34), (175, 238, 238), (255, 248, 220),
             (218, 165,  32), (255, 250, 240), (253, 245, 230), (244, 164,  96), (210, 105,  30)]

def draw_bbox(img, box, cls_name, identity=None, offset=(0, 0)):
    '''
        draw box of an id
    '''
    x1, y1, x2, y2 = [int(i+offset[idx % 2]) for idx, i in enumerate(box)]
    # set color and label text
    color = COLORS_10[identity %
                      len(COLORS_10)] if identity is not None else COLORS_10[0]
    label = '{} {}'.format(cls_name, identity)
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
    cv2.putText(img, label, (x1, y1+t_size[1]+4),
                cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img

def draw_bboxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
        cv2.putText(
            img, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def viz_mcmt(args,cam,cam_track):
    print("start cam:{}".format(cam))
    cam_dir = os.path.join(args.data_path,'test','S06','c0{}'.format(cam))
    cap = cv2.VideoCapture(os.path.join(cam_dir,"vdo.avi"))
    out_dit = os.path.join(args.output_path,'mcmt','S06')
    img_dir = os.path.join(args.output_path,'mcmt','S06','imgs')
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    out = cv2.VideoWriter(os.path.join(out_dit,"c0{}.mp4".format(cam)),
                          cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1920, 1080))
    fr_id = 1
    state,im = cap.read()
    while (state):
        if fr_id in cam_track:
            bbox_xyxy = []
            pids = []
            for pid,x,y,w,h in cam_track[fr_id]:
                bbox_xyxy.append([x, y, x + w, y + h])
                pids.append(pid)
                img_dir_out = os.path.join(img_dir,'{}'.format(pid))
                if not os.path.isdir(img_dir_out):
                    os.makedirs(img_dir_out)
                clip = im[y:(y+h),x:(x+w)]
                cv2.imwrite(os.path.join(img_dir_out,'c0{}_{}_{}.jpg'.format(cam,pid,fr_id)),clip)

            im = draw_bboxes(im, bbox_xyxy, pids)
            cv2.putText(im, "%d" % (fr_id), (0, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 2)
            # plt.imshow(im)
            # plt.show()
        im=cv2.resize(im,(1920,1080))
        out.write(im)
        state, im = cap.read()
        fr_id += 1
    out.release()


def main(args):
    with open(args.mcmt_path,"r") as f:
        mct_tracks = f.readlines()
    cam_tracks = dict()
    for track_line in mct_tracks:
        c,cid,f,x,y,w,h,_,_=tuple([int(float(sstr)) for sstr in track_line.split(' ')])
        if c in cam_tracks:
            if f in cam_tracks[c]:
                cam_tracks[c][f].append((cid, x, y, w, h))
            else:
                cam_tracks[c][f]=[(cid,x,y,w,h)]
        else:
            cam_tracks[c]={f:[(cid,x,y,w,h)]}

    for c in cam_tracks:
        viz_mcmt(args,c,cam_tracks[c])

if __name__ == "__main__":
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    args.data_path = cfg.CHALLENGE_DATA_DIR
    args.output_path = '../../../exp/viz'
    args.mcmt_path = cfg.MCMT_OUTPUT_TXT
    main(args)
