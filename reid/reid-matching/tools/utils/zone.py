import os
from os.path import join as opj
import cv2
import numpy as np

BBOX_B = 10/15

class zone_tracker():
    def __init__(self):
        self.zone_list = list()
        self.id2id = dict()

    def update(self,det):
        have_up = False
        det_time = int(det['frame'][3:])
        det_x = 1280 - (det['bbox'][2]+det['bbox'][0])/2
        det_y = (det['bbox'][3]+det['bbox'][1])/2
        max_sim = 0
        max_zi = 0
        for zi,zl in enumerate(self.zone_list):
            zl_time = int(zl['frame'][3:])
            if det_time==zl_time:
                continue
            if det_time>zl_time+4:
                self.zone_list[zi] = -1
                continue
            if zl['id']==det['id']:
                self.zone_list[zi] = det
                have_up = True
                break
            if det['id'] in self.id2id:
                if self.id2id[det['id']]==zl['id']:
                    if det_time==zl_time:
                        have_up = True
                        break
                    else:
                        det['id'] = zl['id']
                        self.id2id[det['id']] = zl['id']
                        self.zone_list[zi] = det
                        have_up = True
                        break
            zl_x = 1280 - (zl['bbox'][2] + zl['bbox'][0]) / 2
            zl_y = (zl['bbox'][3] + zl['bbox'][1]) / 2
            if det_x<zl_x+10 and det_y<zl_y+10:
                continue
            if abs(det_x-zl_x)<15 and abs(det_y-zl_y)<15:
                # print("det:{}  zl:{}".format([det['id'],det_x,det_y],[zl['id'],zl_x,zl_y]))
                if det['id'] in self.id2id:
                    continue
                det['id'] = zl['id']
                self.id2id[det['id']]=zl['id']
                self.zone_list[zi] = det
                have_up = True
                break
        self.zone_list = [zl for zl in self.zone_list if zl!=-1]
        if not have_up:
            self.zone_list.append(det)
        return det

class zone():
    def __init__(self):
        # 0: b 1: g 3: r 123:w
        # w r 非高速
        # b g 高速
        zones = {}
        zone_path = "./zone"
        for img_name in os.listdir(zone_path):
            camnum = int(img_name.split('.')[0][-3:])
            zone_img = cv2.imread(opj(zone_path, img_name))
            zones[camnum] = zone_img

        self.zones = zones
        self.current_cam = 0

    def set_cam(self,cam):
        self.current_cam = cam

    def get_zone(self,bbox):
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        pix = self.zones[self.current_cam][cy, cx, :]
        zone_num = 0
        if pix[0] > 50 and pix[1] > 50 and pix[2] > 50:  # w
            zone_num = 1
        if pix[0] < 50 and pix[1] < 50 and pix[2] > 50:  # r
            zone_num = 2
        if pix[0] < 50 and pix[1] > 50 and pix[2] < 50:  # g
            zone_num = 3
        if pix[0] > 50 and pix[1] < 50 and pix[2] < 50:  # b
            zone_num = 4
        return zone_num

    def is_ignore(self,zone_list,frame_list, cid):
        # 0 不在任何路口 1 白色 2 红色 3 绿色 4 蓝色
        zs, ze = zone_list[0], zone_list[-1]
        fs, fe = frame_list[0],frame_list[-1]
        if zs == ze:
            # 如果一直在一个区域里，排除
            if ze in [1,2]:
                return True
            if zs!=0 and 0 in zone_list:
                return False
            if fe-fs>1500:
                return True
            if fs<2:
                if cid in [45]:
                    return True
            if fe > 1999:
                if cid in [41]:
                    return True
            if fs<2 or fe>1999:
              if ze in [3,4]:
                return False
            return True
        else:
            # 如果区域发生变化
            if cid in [41, 42, 43, 44, 45, 46]:
                # 如果从支路进支路出，排除
                if zs == 1 and ze == 2:
                    return True
                if zs == 2 and ze == 1:
                    return True
            if cid in [41]:
                # 在41相机，车辆没有进出42相机
                if (zs in [1, 2]) and ze == 4:
                    return True
                if zs == 4 and (ze in [1, 2]):
                    return True
            if cid in [46]:
                # 在46相机，车辆没有进出45相机
                if (zs in [1, 2]) and ze == 3:
                    return True
                if zs == 3 and (ze in [1, 2]):
                    return True
            return False

    def filter_mot(self,mot_list, cid):
        new_mot_list = dict()
        for tracklet in mot_list:
            tracklet_dict = mot_list[tracklet]
            frame_list = list(tracklet_dict.keys())
            frame_list.sort()
            zone_list = []
            for f in frame_list:
                zone_list.append(tracklet_dict[f]['zone'])
            if not self.is_ignore(zone_list,frame_list, cid):
                new_mot_list[tracklet] = tracklet_dict
        return new_mot_list

    def filter_bbox(self,mot_list,cid):
        new_mot_list = dict()
        yh = self.zones[cid].shape[0]
        for tracklet in mot_list:
            # if tracklet==1812:
            #     print(tracklet)
            tracklet_dict = mot_list[tracklet]
            frame_list = list(tracklet_dict.keys())
            frame_list.sort()
            bbox_list = []
            for f in frame_list:
                bbox_list.append(tracklet_dict[f]['bbox'])
            bbox_x = [b[0] for b in bbox_list]
            bbox_y = [b[1] for b in bbox_list]
            bbox_w = [b[2]-b[0] for b in bbox_list]
            bbox_h = [b[3]-b[1] for b in bbox_list]
            new_frame_list = list()
            if 0 in bbox_x or 0 in bbox_y:
                b0 = [i for i, f in enumerate(frame_list) if bbox_x[i]<5 or bbox_y[i]+bbox_h[i]>yh-5]
                if len(b0)==len(frame_list):
                    if cid in [41,42,44,45,46]:
                        continue
                    max_w = max(bbox_w)
                    max_h = max(bbox_h)
                    for i,f in enumerate(frame_list):
                        if bbox_w[i] > max_w * BBOX_B and bbox_h[i] > max_h * BBOX_B:
                            new_frame_list.append(f)
                else:
                    l_i,r_i = 0,len(frame_list)-1
                    if b0[0]==0:
                        for i in range(len(b0)-1):
                            if b0[i]+1==b0[i+1]:
                                l_i=b0[i+1]
                            else:
                                break
                    if b0[-1]==len(frame_list)-1:
                        for i in range(len(b0) - 1):
                            i = len(b0)-1-i
                            if b0[i]-1==b0[i-1]:
                                r_i=b0[i-1]
                            else:
                                break

                    max_lw, max_lh = bbox_w[l_i], bbox_h[l_i]
                    max_rw, max_rh = bbox_w[r_i], bbox_h[r_i]
                    for i,f in enumerate(frame_list):
                        if i < l_i:
                            if bbox_w[i] > max_lw * BBOX_B and bbox_h[i] > max_lh * BBOX_B:
                                new_frame_list.append(f)
                        elif i > r_i:
                            if bbox_w[i] > max_rw * BBOX_B and bbox_h[i] > max_rh * BBOX_B:
                                new_frame_list.append(f)
                        else:
                            new_frame_list.append(f)
                new_tracklet_dict=dict()
                for f in new_frame_list:
                    new_tracklet_dict[f]=tracklet_dict[f]
                new_mot_list[tracklet] = new_tracklet_dict
            else:
                new_mot_list[tracklet] = tracklet_dict
        return new_mot_list

    def break_mot(self,mot_list,cid):
        # if not cid in [41,44,45,46]:
        #     return mot_list
        new_mot_list = dict()
        new_num_tracklets = max(mot_list)+1
        for tracklet in mot_list:
            tracklet_dict = mot_list[tracklet]
            frame_list = list(tracklet_dict.keys())
            frame_list.sort()
            zone_list = []
            back_tracklet = False
            new_zone_f = 0
            pre_frame = frame_list[0]
            time_break = False
            # if tracklet == 714:
            #     print(tracklet)
            for f in frame_list:
                if f-pre_frame>100:
                    if cid in [44,45]:
                        time_break = True
                        break
                if not cid in [41, 44, 45, 46]:
                    break
                pre_frame=f
                new_zone = tracklet_dict[f]['zone']
                if len(zone_list)>0 and zone_list[-1] == new_zone:
                    continue
                if new_zone_f>1:
                    if len(zone_list) > 1 and new_zone in zone_list:
                        back_tracklet = True
                    zone_list.append(new_zone)
                    new_zone_f=0
                else:
                    new_zone_f+=1
            if back_tracklet:
                new_tracklet_dict = dict()
                pre_bbox = -1
                pre_arrow = 0
                have_break = False
                for f in frame_list:
                    now_bbox = tracklet_dict[f]['bbox']
                    if pre_bbox == -1:
                        pre_bbox = now_bbox
                    now_arrow = now_bbox[0] - pre_bbox[0]
                    if pre_arrow*now_arrow<0 and len(new_tracklet_dict)>15 and not have_break:
                        new_mot_list[tracklet] = new_tracklet_dict
                        new_tracklet_dict = dict()
                        have_break = True
                    if have_break:
                        tracklet_dict[f]['id'] = new_num_tracklets
                    new_tracklet_dict[f] = tracklet_dict[f]
                    pre_bbox,pre_arrow = now_bbox,now_arrow
                if have_break:
                    new_mot_list[new_num_tracklets] = new_tracklet_dict
                    new_num_tracklets += 1
                else:
                    new_mot_list[tracklet] = new_tracklet_dict
            elif time_break:
                new_tracklet_dict = dict()
                have_break = False
                pre_frame = frame_list[0]
                for f in frame_list:
                    if f-pre_frame>100:
                        new_mot_list[tracklet] = new_tracklet_dict
                        new_tracklet_dict = dict()
                        have_break = True
                    new_tracklet_dict[f] = tracklet_dict[f]
                    pre_frame=f
                if have_break:
                    new_mot_list[new_num_tracklets] = new_tracklet_dict
                    new_num_tracklets += 1
                else:
                    new_mot_list[tracklet] = new_tracklet_dict
            else:
                new_mot_list[tracklet] = tracklet_dict
        print("old:{} new:{}".format(len(mot_list),len(new_mot_list)))
        return new_mot_list

    def comb_mot(self,mot_list,cid):
        time_mot = dict()
        for tid in mot_list:
            tracklet = mot_list[tid]
            for tf in tracklet:
                if tf in time_mot:
                    time_mot[tf].append(tracklet[tf])
                else:
                    time_mot[tf]=[tracklet[tf]]
        time_list = list(time_mot)
        time_list.sort()
        zone_tack = zone_tracker()
        new_mot_list = dict()
        for t in time_list:
            det_list = time_mot[t]
            for di,dt in enumerate(det_list):
                if dt['zone'] in [4]:
                    dt = zone_tack.update(dt)
                else:
                    if dt['id'] in zone_tack.id2id:
                        dt['id']=zone_tack.id2id[dt['id']]
                time_mot[t][di] = dt
                if dt['id'] in new_mot_list:
                    new_mot_list[dt['id']][int(dt['frame'][3:])]=dt
                else:
                    new_mot_list[dt['id']] = {int(dt['frame'][3:]): dt}
        print("old:{} new:{}".format(len(mot_list),len(new_mot_list)))
        return new_mot_list