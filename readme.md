# [City-Scale Multi-Camera Vehicle Tracking Guided by Crossroad Zones](https://arxiv.org/pdf/2105.06623.pdf)

## Requirements

Python 3.8 or later with all ```requirements.txt``` dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```

## Data Preparation
If you want to reproduce our results on AI City Challengef,
please download the datasets from: (https://www.aicitychallenge.org/)
and put it under the folder datasets.
Make sure the data structure is like:

> **[AIC21-MTMC](https://drive.google.com/drive/folders/11616Gomc7MbjbgWrDruL26TGi9JNCAAE?usp=sharing)**
>   * datasets
>     * [AIC21_Track3_MTMC_Tracking](https://www.aicitychallenge.org/2021-data-and-evaluation/)
>       * unzip AIC21_Track3_MTMC_Tracking.zip
>     * detect_provided (Including detection and corresponding Re-ID features)
>   * detector
>     * yolov5
>       * [yolov5x.pt](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5x.pt) (Pre-trained yolov5x model on COCO)
>   * reid
>     * reid_model (Pre-trained reid model on Track 2)
>       * resnet101_ibn_a_2.pth
>       * resnet101_ibn_a_3.pth
>       * resnext101_ibn_a_2.pth

## Reproduce frome detect_provided 
If you just want reproduce our results, you can directly download ```detect_provided```:
```
cd AIC21-MTMC
mkdir datasets
cd datasets
```
Then put ```detect_provided``` folder under this folder and modify yml ```config/aic_mcmt.yml```:
```
CHALLENGE_DATA_DIR: '/home/xxx/AIC21-MTMC/datasets/AIC21_Track3_MTMC_Tracking/'
DET_SOURCE_DIR: '/home/xxx/AIC21-MTMC/datasets/detection/images/test/S06/'
DATA_DIR: '/home/xxx/AIC21-MTMC/datasets/detect_provided'
REID_SIZE_TEST: [384, 384]
ROI_DIR: '/home/xxx/AIC21-MTMC/datasets/AIC21_Track3_MTMC_Tracking/test/S06/'
CID_BIAS_DIR: '/home/xxx/AIC21-MTMC/datasets/AIC21_Track3_MTMC_Tracking/cam_timestamp/'
USE_RERANK: True
USE_FF: True
SCORE_THR: 0.1
MCMT_OUTPUT_TXT: 'track3.txt'
```
Then run:
```
bash ./run_mcmt.sh
```

The final results will locate at path ```./reid/reid-matching/tools/track3.txt```

## Reproduce on all pipeline
If you just want reproduce our results on all pipeline, you have to download:
```
detector/yolov5/yolov5x.pt
reid/reid_model/resnet101_ibn_a_2.pth
reid/reid_model/resnet101_ibn_a_3.pth
reid/reid_model/resnext101_ibn_a_2.pth
```
You can refer to [Track2](https://github.com/michuanhaohao/AICITY2021_Track2_DMT) to retrain the reid model.

Then modify yml:
```
config/aic_all.yml
config/aic_reid1.yml
config/aic_reid2.yml
config/aic_reid3.yml
```
Then run:
```
bash ./run_all.sh
```

The final results will locate at path ```./reid/reid-matching/tools/track3.txt```
