import os
import sys
sys.path.append('../')
from config import cfg
import cv2
from tqdm import tqdm

def preprocess(src_root, dst_root):
    if not os.path.isdir(src_root):
        print("[Err]: invalid source root")
        return

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)
        print("{} made".format(dst_root))

    sec_dir_list = ['test']
    dst_dir_list = [ dst_root + '/images/' + i for i in sec_dir_list]
    for i in dst_dir_list:
        if not os.path.isdir(i):
            os.makedirs(i)

    for i,x in enumerate(sec_dir_list):
        x_path = src_root + '/' + x
        if os.path.isdir(x_path):
            for y in os.listdir(x_path):
                if y.startswith('S'):
                    y_path = os.path.join(x_path,y)
                    for z in os.listdir(y_path):
                        z_path = os.path.join(y_path,z)
                        if z.startswith('c'):
                            video_path = os.path.join(z_path,'vdo.avi')
                            roi_path = os.path.join(z_path, 'roi.jpg')
                            ignor_region = cv2.imread(roi_path)

                            dst_img1_dir = os.path.join(dst_dir_list[i],y,z,'img1')
                            if not os.path.isdir(dst_img1_dir):
                                os.makedirs(dst_img1_dir)

                            # 生成图片帧
                            video = cv2.VideoCapture(video_path)
                            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                            frame_current = 0
                            while frame_current<frame_count-1:
                                frame_current = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                                _, frame = video.read()
                                dst_f =  'img{:06d}.jpg'.format(frame_current)
                                dst_f_path = os.path.join(dst_img1_dir , dst_f)
                                if not os.path.isfile(dst_f_path):
                                    frame = draw_ignore_regions(frame, ignor_region)
                                    cv2.imwrite(dst_f_path, frame)
                                    print('{}:{} generated to {}'.format(z,dst_f, dst_img1_dir))
                                else:
                                    print('{}:{} already exists.'.format(z,dst_f))

def draw_ignore_regions(img, region):
    if img is None:
        print('[Err]: Input image is none!')
        return -1
    img = img*(region>0)

    return img
if __name__ == '__main__':
    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    save_dir = cfg.DET_SOURCE_DIR.split('images')[0]
    preprocess(src_root=f'{cfg.CHALLENGE_DATA_DIR}',
               dst_root=f'{save_dir}')
    print('Done')
