import lap  # linear assignment problem solver using Jonker-Volgenant algorithm
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from .tracking_utils import KalmanFilter

# STrack : definition - basetrack.py

def embedding_distance(tracks, detections, metric='cosine'):
    #tracks : list of STrack
    #detections : list of BaseTrack
    #return : cost_matrix np.ndarray

    cost_matrix = np.zeros((len(tracks),len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  #cdist : sqrt( sum((a-b)**2, axis=1)) : sqrt ( square the difference between each element)
    return cost_matrix

def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    # using kalman filter 
    # lambda_ : ctrl 
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = KalmanFilter().chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0,2), dtype=np.int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def iou(atlbrs, btlbrs):
    """
    IOU calculate
    atlbrs : list[tlbr] (type : np.ndarray) [[top, left, bottom, right]]
    btlbrs : list[tlbr] (type : np.ndarray)
    result : np.ndarray
    """

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)

    if ious.size == 0:
        return ious
    
    x1 = np.zeros((len(atlbrs),len(btlbrs)), dtype=np.float)
    y1 = np.zeros((len(atlbrs),len(btlbrs)), dtype=np.float)
    x2 = np.zeros((len(atlbrs),len(btlbrs)), dtype=np.float)
    y2 = np.zeros((len(atlbrs),len(btlbrs)), dtype=np.float)

    for a in range(len(atlbrs)):
        for b in range(len(btlbrs)):
            x1[a][b] = max(atlbrs[a][0], btlbrs[b][0])
            y1[a][b] = max(atlbrs[a][1], btlbrs[b][1])
            x2[a][b] = min(atlbrs[a][2], btlbrs[b][2])
            y2[a][b] = min(atlbrs[a][3], btlbrs[b][3])

    area_intersection = (x2 - x1 + 1) * (y2 - y1 + 1)
    areas_a = (atlbrs[..., 2] - atlbrs[..., 0] + 1) * (atlbrs[..., 3] - atlbrs[..., 1] + 1)
    areas_b = (btlbrs[..., 2] - btlbrs[..., 0] + 1) * (btlbrs[..., 3] - btlbrs[..., 1] + 1)
    area_union = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    for a in range(len(atlbrs)):
        for b in range(len(btlbrs)):
            area_union[a][b] = areas_a[a] + areas_b[b] - area_intersection[a][b]
    
    ious = area_intersection / area_union
    
    return ious

def iou_distance(atracks, btracks):
    if (len(atracks)> 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        atlbrs = np.array(atlbrs)
        btlbrs = np.array(btlbrs)

    _iou = iou(atlbrs, btlbrs)
    cost_matrix = 1 - _iou

    return cost_matrix

