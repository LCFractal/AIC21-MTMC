import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda import amp
import torch.distributed as dist
import logging
import os
from utils.meter import AverageMeter
from utils.metrics import R1_mAP, R1_mAP_eval, R1_mAP_Pseudo, R1_mAP_query_mining


def do_uda_train(epoch, cfg, model, center_creterion, train_loader, val_loader, optimizer, optimizer_center, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    scaler = amp.GradScaler()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    device = 'cuda'
    logger = logging.getLogger('reid_baseline.train')

    model.train()
    #model_ema.train()
    for n_iter in range(300):
        img, vid, target_cam = train_loader.next()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img = img.to(device)
        target = vid.to(device)
        target_cam = target_cam.to(device)

        if cfg.SOLVER.FP16_ENABLED:
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam)
                loss = loss_fn(score, feat, target, target_cam)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            score, feat = model(img, target, cam_label=target_cam)
            loss = loss_fn(score, feat, target, target_cam)
            loss.backward()
            optimizer.step()

        def update_ema_variables(model, ema_model, global_step):
            alpha = 0.999
            alpha = min(1 - 1 / (global_step + 1), alpha)
            for (ema_name, ema_param), (model_name, model_param) in zip(ema_model.named_parameters(), model.named_parameters()):
                ema_param.data = ema_param.data * alpha + model_param.data * (1 - alpha)
        
        #update_ema_variables(model, model_ema, epoch*len(train_loader) + n_iter)

        if isinstance(score, list):
            acc = (score[0].max(1)[1] == target).float().mean()
        else:
            acc = (score.max(1)[1] == target).float().mean()
        loss_meter.update(loss.item(), img.shape[0])
        acc_meter.update(acc, 1)

        torch.cuda.synchronize()

        if (n_iter + 1) % log_period == 0:
            logger.info('Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}'.format(epoch, (n_iter+1), 300, loss_meter.avg, acc_meter.avg))

    if (epoch + 1) % checkpoint_period == 0:
        if cfg.MODEL.DIST_TRAIN:
            if dist.get_rank() == 0:
                torch.save(model.module.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        else:
            torch.save(model.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

    if (epoch + 1) % eval_period == 0:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, dataset = cfg.DATASETS.NAMES)
        evaluator.reset()
        if cfg.MODEL.DIST_TRAIN:
            if dist.get_rank() == 0:
                model_ema.eval()
                torch.cuda.empty_cache()
                for n_iter, (img, vid, camid, trackids, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = torch.tensor(camid, dtype=torch.int64)
                        camids = camids.to(device)
                        feat = model_ema(img, cam_label=camids)
                        evaluator.update((feat.clone(), vid, camid, trackids))

                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info(cfg.OUTPUT_DIR)
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
        else:
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for n_iter, (img, vid, camid, trackids, _) in enumerate(val_loader):
                    img = img.to(device)
                    camids = torch.tensor(camid, dtype=torch.int64)
                    camids = camids.to(device)
                    feat = model(img, cam_label=camids)
                    evaluator.update((feat.clone(), vid, camid, trackids))

            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info(cfg.OUTPUT_DIR)
            logger.info("Validation Results Standard Model - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

            #model_ema.eval()
            #torch.cuda.empty_cache()
            #with torch.no_grad():
            #    for n_iter, (img, vid, camid, trackids, _) in enumerate(val_loader):
            #        img = img.to(device)
            #        camids = torch.tensor(camid, dtype=torch.int64)
            #        camids = camids.to(device)
            #        feat = model_ema(img, cam_label=camids)
            #        evaluator.update((feat.clone(), vid, camid, trackids))

            #cmc, mAP, _, _, _, _, _ = evaluator.compute()
            #logger.info(cfg.OUTPUT_DIR)
            #logger.info("Validation Results Ema Model - Epoch: {}".format(epoch))
            #logger.info("mAP: {:.1%}".format(mAP))
            #for r in [1, 5, 10]:
            #    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            #torch.cuda.empty_cache()
