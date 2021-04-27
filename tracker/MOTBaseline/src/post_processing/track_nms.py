import numpy as np

def remove_1len_track(all_results):
    refined_results = []
    tids = np.unique(all_results[:, 1])
    for tid in tids:
        results = all_results[all_results[:, 1] == tid]
        if results.shape[0] <= 1:
            continue
        refined_results.append(results)

    refined_results = np.concatenate(refined_results, axis=0)

    return refined_results

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b.T)


def noverlap(period1, period2):
    s1 = period1[0]
    e1 = period1[-1]
    s2 = period2[0]
    e2 = period2[-1]

    if (0 < s2 - e1) or (0 < s1 - e2):
        return True

    return False

def reid_similarity(det1, det2):
    feat1 = det1[:, 8:]
    feat2 = det2[:, 8:]
    avg_feat1 = np.mean(feat1, axis=0)
    avg_feat2 = np.mean(feat2, axis=0)
    return cosine_similarity(avg_feat1, avg_feat2)

def associate(det, threshold):
    processed_track_list = []
    match = {}
    tids = np.unique(det[:, 1])
    for i in range(len(tids) - 1):
        if i in processed_track_list:
            continue
        for j in range(i+1, len(tids)):
            if j in processed_track_list:
               continue
            det_i = det[det[:, 1] == tids[i]]
            det_j = det[det[:, 1] == tids[j]]
            image_ids_i = det_i[:, 0]
            image_ids_j = det_j[:, 0]
            if noverlap(image_ids_i, image_ids_j):
                similarity = reid_similarity(det_i, det_j)
                if similarity > threshold:
                    match[tids[j]] =  tids[i]
                    processed_track_list.append(j)
    if len(match) != 0:
        results = []
        for tid in tids:
            sub_det = det[det[:, 1] == tid]
            if tid in match:
                sub_det[:, 1] = match[tid]
            results.append(sub_det)
        det = np.vstack(results)
    return det

def isoverlaped(det1, det2):
    is_overlap = False
    for line1 in det1:
        for line2 in det2:
            if line1[0] == line2[0]:
                is_overlap = True
                i, u = bb_intersect_union(line1[2:6], line2[2:6])
                iou = i / u if u > 0 else 0
                #print(iou)
                if iou < 0.99:
                    return False
    return is_overlap

def associate_overlap_track(det):
    match = []
    tids = np.unique(det[:, 1])
    for i in range(len(tids)-1):
        for j in range(i+1, len(tids)):
            det_i = det[det[:, 1] == tids[i]].copy()
            det_j = det[det[:, 1] == tids[j]].copy()
            image_ids_i = det_i[:, 0]
            image_ids_j = det_j[:, 0]
            if not noverlap(image_ids_i, image_ids_j):
                if isoverlaped(det_i, det_j):
                    match.append([tids[i], tids[j]])
    #if len(match) > 0:
    #    import pdb
    #    pdb.set_trace()
    processed_match = []
    agg_match = []
    for i in range(len(match)-1):
        if i in processed_match:
            continue
        match_i = set(match[i])
        for j in range(i+1, len(match)):
            if j in processed_match:
                continue
            if len(match_i & set(match[j])) > 0:
                match_i = match_i | set(match[j])
                processed_match.append(j)
        agg_match.append(list(match_i))

    if len(agg_match) != 0:
        results = []
        total_matched_tids = []
        for m in agg_match:
            total_matched_tids += m
            res = []
            for tid in m:
                sub_det = det[det[:, 1] == tid].copy()
                sub_det[:, 1] = m[0]
                res.append(sub_det)
            res = np.vstack(res)
            iids, index = np.unique(res[:,0].astype(int), return_index=True)
            res = res[index]
            results.append(res)

        for tid in tids:
            if tid not in total_matched_tids:
                sub_det = det[det[:, 1] == tid].copy()
                results.append(sub_det)

        det = np.vstack(results)
    return det


def bb_intersect_union(bbox1, bbox2):
    dx, dy, dw, dh = bbox1
    gx, gy, gw, gh = bbox2

    d_area = dw * dh
    g_area = gw * gh

    left = max(dx, gx)
    right = min(dx + dw, gx + gw)
    top = max(dy, gy)
    bottom = min(dy + dh, gy + gh)

    w = max(right - left, 0)
    h = max(bottom - top, 0)

    i = w * h
    u = d_area + g_area - i
    return i, u

def iou_3d(track1, track2):
    u = 0
    i = 0
    image_ids = set(track1.keys()) | set(track2.keys())
    for iid in image_ids:
        bbox1 = track1.get(iid, None)
        bbox2 = track2.get(iid, None)
        if bbox1 is not None and bbox2 is not None:
            i_, u_ = bb_intersect_union(bbox1, bbox2)
            i += i_
            u += u_
        elif bbox1 is None and bbox2 is not None:
            u += bbox2[2] * bbox2[3]
        elif bbox1 is not None and bbox2 is None:
            u += bbox1[2] * bbox1[3]

    assert i <= u
    return i / u if u > 0 else 0

def ioshort_3d(track1, track2):
    assert len(set(track2.keys())) >= len(set(track1.keys()))
    u = 0
    i = 0
    if len(set(track1.keys()) | set(track2.keys())) < 5:
        return 0
    image_ids = set(track1.keys()) | set(track2.keys())
    for iid in image_ids:
        bbox1 = track1.get(iid, None)
        bbox2 = track2.get(iid, None)
        if bbox1 is not None and bbox2 is not None:
            i_, u_ = bb_intersect_union(bbox1, bbox2)
            i += i_
            u += u_
        # elif bbox1 is None and bbox2 is not None:
        #     u += bbox2[2] * bbox2[3]
        elif bbox1 is not None and bbox2 is None:
            u += bbox1[2] * bbox1[3]

    assert i <= u
    return i / u if u > 0 else 0

def nms_3d(tracks, length, threshold):
    length = np.array(length)
    order = length.argsort()[::-1]

    keep = []
    while order.size > 0:
        keep.append(order[0])
        ious = []
        for i in range(1, order.size):
            #iou = iou_3d(tracks[order[i]], tracks[order[0]])
            iou = ioshort_3d(tracks[order[i]], tracks[order[0]])
            ious.append(iou)
        ious = np.array(ious)

        inds = np.where(ious <= threshold)[0]
        order = order[inds + 1]
    return keep


def track_nms(tracks, nms_thre=0.65):
    tracks_dict = {} # transfer to dict format
    for row in tracks:
        frame_idx = int(row[0])
        track_id = int(row[1])
        if track_id not in tracks_dict:
            tracks_dict[track_id] = {frame_idx : row[2:6]}
        else:
            tracks_dict[track_id][frame_idx] = row[2:6]
    tracks_dict = list(tracks_dict.values()) # store tracks as track-nms format
    length_list = [len(trk) for trk in tracks_dict] # track length as 3d nms score
    keep = nms_3d(tracks_dict, length_list, nms_thre)
    trk_ids = np.unique(tracks[:, 1])
    print('after track nms, removing ', (len(trk_ids) - len(keep)) ,' tracks')
    valid_ids = trk_ids[keep]
    tracks = np.array([row for row in tracks if row [1] in valid_ids])
    return tracks
