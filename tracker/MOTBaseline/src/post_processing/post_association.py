import numpy as np
import os

INFTY_COST = 1e+5
BBOX_X = 400
BBOX_Y = 300

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b.T)

def noverlap(period1, period2):
    image_ids_1 = period1[:, 0]
    image_ids_2 = period2[:, 0]
    s_bbox1 = period1[np.argmin(image_ids_1), 2:6]
    e_bbox1 = period1[np.argmax(image_ids_1), 2:6]
    s_bbox2 = period2[np.argmin(image_ids_2), 2:6]
    e_bbox2 = period2[np.argmax(image_ids_2), 2:6]
    s_cx1,s_cy1 = s_bbox1[0]+s_bbox1[2]/2,s_bbox1[1]+s_bbox1[3]/2
    e_cx1,e_cy1 = e_bbox1[0]+e_bbox1[2]/2,e_bbox1[1]+e_bbox1[3]/2
    s_cx2,s_cy2 = s_bbox2[0]+s_bbox2[2]/2,s_bbox2[1]+s_bbox2[3]/2
    e_cx2,e_cy2 = e_bbox2[0]+e_bbox2[2]/2,e_bbox2[1]+e_bbox2[3]/2

    s1 = np.min(image_ids_1)
    e1 = np.max(image_ids_1)
    s2 = np.min(image_ids_2)
    e2 = np.max(image_ids_2)

    if abs(s_cy2-e_cy1)>BBOX_Y or abs(s_cx2-e_cx1)>BBOX_X:
        return False

    if abs(s_cy1-e_cy2)>BBOX_Y or abs(s_cx1-e_cx2)>BBOX_X:
        return False

    if (50 < s2 - e1) or (50 < s1 - e2):
        return False

    if (0 < s2 - e1) or (0 < s1 - e2):
        return True
     
    return False

def reid_similarity(det1, det2, start_cols):
    feat1 = det1[:, start_cols:]
    #print(feat1.shape)
    assert feat1.shape[1] == 128 or feat1.shape[1] == 512 or feat1.shape[1] == 2048
    feat2 = det2[:, start_cols:]
    avg_feat1 = np.mean(feat1, axis=0)
    avg_feat2 = np.mean(feat2, axis=0)
    return cosine_similarity(avg_feat1, avg_feat2)

def associate(det, threshold, start_cols, seq):
    #processed_track_list = []
    tids = np.unique(det[:, 1])
    cost_m = np.ones( (len(tids), len(tids))) * threshold
    edges = []
    for i in range(len(tids) - 1):
        trk_i = det[det[:, 1] == tids[i]]
        # image_ids_i = trk_i[:, 0]
        # ignore len 1 track
        #if (trk_i.shape[0] == 1): continue
        for j in range(i+1, len(tids)):
            trk_j = det[det[:, 1] == tids[j]]
            #if (trk_j.shape[0] == 1): continue
            # image_ids_j = trk_j[:, 0]
            if noverlap(trk_i, trk_j):
                similarity = reid_similarity(trk_i, trk_j, start_cols)
                cost_m[i,j] = similarity
                if similarity < threshold:
                    #print(tids[i], tids[j])
                    edges.append([i, j, similarity])
                #     match[tids[j]] =  tids[i]
                #     processed_track_list.append(j)
    if not os.path.exists('./cache/'):
        os.makedirs('./cache/')
    of = open(f'./cache/{seq}_PA_cost.txt', 'w')
    print(len(tids), file=of)
    print(len(edges), file=of)
    for edge in edges:
        print(edge[0], edge[1], edge[2], file=of)
    of.close()
    print('generating input graph for solving min cost perfect matching problem.')
    os.system(f'./MinCostPerfMatch/example -f ./cache/{seq}_PA_cost.txt --max > ./cache/{seq}_PA_res.txt')
    matches = np.loadtxt(f'./cache/{seq}_PA_res.txt', dtype=int, delimiter=' ')
    print(f'in this stage of PA, get matched tracklets {len(matches)}')
    if len(matches) != 0:
        if len(matches.shape) == 1: matches = np.array([matches,])
        matching = {tids[match[0]]: tids[match[1]] for match in matches}
        results = []
        for tid in tids:
            sub_det = det[det[:, 1] == tid]
            if tid in matching:
                sub_det[:, 1] = matching[tid]
            results.append(sub_det)
        det = np.vstack(results)
    return det

