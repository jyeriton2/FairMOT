import torch
from torch import nn
from .util import _gather_feat, _transpose_and_gather_feat

def _nms(heat, kernel=3):
    pad = (kernel -1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    '''
    if hmax == heat:
        keep = 1.0
    else:
        keep = 0.0
    '''

    return heat*keep

def _topk_channel(scores, K):    # call top K
    batch, cat, height, width = scores.size()

    topk_scores, topk_idx = torch.topk(scores.view(batch, cat, -1), K) #.view(...,-1) dimension -1 
    topk_idx = topk_idx % (height * width)
    topk_ys = (topk_idx / width).int().float()
    topk_xs = (topk_idx % width).int().float()

    return topk_scores, topk_idx, topk_ys, topk_xs

def _topk(scores, K):
    batch, cat, height, width = scores.size()

    topk_scores, topk_idxs = torch.topk(scores.view(batch, cat, -1), K)

    topk_idxs = topk_idxs % (height * width)
    #topk_ys   = (topk_idxs / width).int().float()
    topk_ys   = (torch.true_divide(topk_idxs, width)).int().float()
    topk_xs   = (topk_idxs % width).int().float()

    topk_score, topk_idx = torch.topk(topk_scores.view(batch, -1), K)
    #topk_classes = (topk_idx / K).int()
    topk_classes = (torch.true_divide(topk_idx, K)).int()
    topk_idxs = _gather_feat(topk_idxs.view(batch, -1, 1), topk_idx).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_idx).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_idx).view(batch, K)

    return topk_score, topk_idxs, topk_classes, topk_ys, topk_xs

def mot_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    heat = _nms(heat)

    scores, idxs, classes, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, idxs)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    wh = _transpose_and_gather_feat(wh, idxs)

    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        class_idx = classes.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, class_idx).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)

    classes = classes.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] /2], dim=2)
    detect = torch.cat([bboxes, scores, classes], dim=2)

    return detect, idxs

