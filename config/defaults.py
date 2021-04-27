from yacs.config import CfgNode as CN

_C = CN()

_C.CHALLENGE_DATA_DIR = ''
_C.DET_SOURCE_DIR = ''
_C.REID_MODEL = ''
_C.REID_BACKBONE = ''
_C.REID_SIZE_TEST = [256, 256]

_C.DET_IMG_DIR = ''
_C.DATA_DIR = ''
_C.ROI_DIR = ''
_C.CID_BIAS_DIR = ''

_C.USE_RERANK = False
_C.USE_FF = False
_C.SCORE_THR = 0.5

_C.MCMT_OUTPUT_TXT = ''
