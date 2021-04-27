"""Extract image feature for both det/mot image feature."""

import os
import pickle
import time
from glob import glob
from itertools import cycle
from multiprocessing import Pool, Queue
import tqdm

import torch
from PIL import Image
import torchvision.transforms as T
from reid_inference.reid_model import build_reid_model
import sys
sys.path.append('../')
from config import cfg

BATCH_SIZE = 64
NUM_PROCESS = 8
def chunks(l):
    return [l[i:i+BATCH_SIZE] for i in range(0, len(l), BATCH_SIZE)]

class ReidFeature():
    """Extract reid feature."""

    def __init__(self, gpu_id, _mcmt_cfg):
        print("init reid model")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.model, self.reid_cfg = build_reid_model(_mcmt_cfg)
        device = torch.device('cuda')
        self.model = self.model.to(device)
        self.model.eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.val_transforms = T.Compose([T.Resize(self.reid_cfg.INPUT.SIZE_TEST, interpolation=3),\
                              T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def extract(self, img_path_list):
        """Extract image feature with given image path.
        Feature shape (2048,) float32."""

        img_batch = []
        for img_path in img_path_list:
            img = Image.open(img_path).convert('RGB')
            img = self.val_transforms(img)
            img = img.unsqueeze(0)
            img_batch.append(img)
        img = torch.cat(img_batch, dim=0)

        with torch.no_grad():
            img = img.to('cuda')
            flip_feats = False
            if self.reid_cfg.TEST.FLIP_FEATS == 'yes': flip_feats = True
            if flip_feats:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                        feat1 = self.model(img)
                    else:
                        feat2 = self.model(img)
                feat = feat2 + feat1
            else:
                feat = self.model(img)
        feat = feat.cpu().detach().numpy()
        return feat


def init_worker(gpu_id, _cfg):
    """init worker."""

    # pylint: disable=global-variable-undefined
    global model
    model = ReidFeature(gpu_id.get(), _cfg)


def process_input_by_worker_process(image_path_list):
    """Process_input_by_worker_process."""

    reid_feat_numpy = model.extract(image_path_list)
    feat_dict = {}
    for index, image_path in enumerate(image_path_list):
        feat_dict[image_path] = reid_feat_numpy[index]
    return feat_dict


def load_all_data(data_path):
    """Load all mode data."""

    image_list = []
    for cam in os.listdir(data_path):
        image_dir = os.path.join(data_path, cam, 'dets')
        cam_image_list = glob(image_dir+'/*.png')
        cam_image_list = sorted(cam_image_list)
        print(f'{len(cam_image_list)} images for {cam}')
        image_list += cam_image_list
    print(f'{len(image_list)} images in total')
    return image_list


def save_feature(output_path, data_path, pool_output):
    """Save feature."""
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    all_feat_dic = {}
    for cam in os.listdir(data_path):
        dets_pkl_file = os.path.join(data_path, cam, f'{cam}_dets.pkl')
        det_dic = pickle.load(open(dets_pkl_file, 'rb'))
        all_feat_dic[cam] = det_dic.copy()

    for sample_dic in pool_output:
        for image_path, feat in sample_dic.items():
            cam = image_path.split('/')[-3]
            image_name = image_path.split('/')[-1].split('.')[0]
            all_feat_dic[cam][image_name]['feat'] = feat
    for cam, feat_dic in all_feat_dic.items():
        if not os.path.isdir(os.path.join(output_path, cam)):
            os.makedirs(os.path.join(output_path, cam))
        feat_pkl_file = os.path.join(output_path, cam, f'{cam}_dets_feat.pkl')
        pickle.dump(feat_dic, open(feat_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('save pickle in %s' % feat_pkl_file)


def extract_image_feat(_cfg):
    """Extract reid feat for each image, using multiprocessing."""

    image_list = load_all_data(_cfg.DET_IMG_DIR)
    chunk_list = chunks(image_list)

    num_process = NUM_PROCESS
    gpu_ids = Queue()
    gpu_id_cycle_iterator = cycle(range(0, 8))
    for _ in range(num_process):
        gpu_ids.put(next(gpu_id_cycle_iterator))

    process_pool = Pool(processes=num_process, initializer=init_worker, initargs=(gpu_ids, _cfg, ))
    start_time = time.time()
    pool_output = list(tqdm.tqdm(process_pool.imap_unordered(\
                                 process_input_by_worker_process, chunk_list),
                                 total=len(chunk_list)))
    process_pool.close()
    process_pool.join()

    # global model
    # model = ReidFeature(0)
    # for sub_list in chunk_list:
    #     ret = process_input_by_worker_process(sub_list)
    print('%.4f s' % (time.time() - start_time))

    save_feature(_cfg.DATA_DIR, _cfg.DET_IMG_DIR, pool_output)


def debug_reid_feat(_cfg):
    """Debug reid feature to make sure the same with Track2."""

    exp_reidfea = ReidFeature(0, _cfg)
    feat = exp_reidfea.extract(['000001.jpg', '000001.jpg', '000001.jpg', '000001.jpg', '000001.jpg'])
    print(feat)


def main():
    """Main method."""

    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    # debug_reid_feat(cfg)
    extract_image_feat(cfg)


if __name__ == "__main__":
    main()
