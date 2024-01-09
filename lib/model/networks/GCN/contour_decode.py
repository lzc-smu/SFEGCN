import torch.nn as nn
import torch


def nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()

    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def decode_ct_hm(ct_hm, wh, reg=None, K=100):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)
    scores, inds, clses, ys, xs = topk(ct_hm, K=K)
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detection = torch.cat([bboxes, scores, clses], dim=2)   # (3,100,6)
    return ct, detection


def gaussian_radius(height, width, min_overlap=0.7):
    height = torch.ceil(height)
    width = torch.ceil(width)

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1.pow(2) - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2.pow(2) - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height

    r31 = torch.min(r1, r2)
    det = b3.pow(2) - 4 * a3 * c3
    sq3 = torch.sqrt(torch.clamp(det, min=0))
    r32 = (b3 + sq3) / 2
    r3_01 = (det < 0).float()
    r3 = r3_01 * r31 + (1 - r3_01) * r32

    radius = torch.clamp(torch.min(torch.min(r1, r2), r3), min=0) / 3
    return torch.round(radius).long()


def get_quadrangle(box):
    x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    quadrangle = [
        (x_min + x_max) / 2., y_min,
        x_min, (y_min + y_max) / 2.,
        (x_min + x_max) / 2., y_max,
        x_max, (y_min + y_max) / 2.
    ]
    quadrangle = torch.stack(quadrangle, dim=2).view(x_min.size(0), x_min.size(1), 4, 2)
    return quadrangle


def get_init(box):
    return get_quadrangle(box)


def decode_ext_hm(ct_hm, ext, ae=None, K=100):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)

    scores, inds, clses, ys, xs = topk(ct_hm, K=K)
    ext = transpose_and_gather_feat(ext, inds)
    ext = ext.view(batch, K, 4, 2)

    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)

    extreme_point = ct[:, :, None] + ext

    xy_min, ind = torch.min(extreme_point, dim=2)
    l_ind, t_ind = ind[..., 0:1], ind[..., 1:2]
    l_ind = l_ind[..., None].expand(l_ind.size(0), l_ind.size(1), 1, 2)
    t_ind = t_ind[..., None].expand(t_ind.size(0), t_ind.size(1), 1, 2)
    ll = extreme_point.gather(2, l_ind)
    tt = extreme_point.gather(2, t_ind)

    xy_max, ind = torch.max(extreme_point, dim=2)
    r_ind, b_ind = ind[..., 0:1], ind[..., 1:2]
    r_ind = r_ind[..., None].expand(r_ind.size(0), r_ind.size(1), 1, 2)
    b_ind = b_ind[..., None].expand(b_ind.size(0), b_ind.size(1), 1, 2)
    rr = extreme_point.gather(2, r_ind)
    bb = extreme_point.gather(2, b_ind)

    extreme_point = torch.cat([tt, ll, bb, rr], dim=2)
    bboxes = torch.cat([xy_min, xy_max], dim=2)
    detection = torch.cat([bboxes, scores, clses], dim=2)

    if ae is not None:
        ae = transpose_and_gather_feat(ae, inds)
        detection = torch.cat([detection, ae], dim=2)

    return ct, extreme_point, detection
