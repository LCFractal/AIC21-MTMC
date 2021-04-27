import os
from .baseline.config import cfg
from .baseline.model import make_model


def build_reid_model(_mcmt_cfg):
    abs_file = __file__
    print("abs path is %s" % (__file__))
    abs_dir = abs_file[:abs_file.rfind("/")]
    cfg.merge_from_file(os.path.join(abs_dir,'aictest.yml'))
    cfg.INPUT.SIZE_TEST = _mcmt_cfg.REID_SIZE_TEST
    cfg.MODEL.NAME = _mcmt_cfg.REID_BACKBONE
    model = make_model(cfg, num_class=100)
    model.load_param(_mcmt_cfg.REID_MODEL)

    return model,cfg
