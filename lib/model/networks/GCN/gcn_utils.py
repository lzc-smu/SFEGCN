import torch
import math
import numpy as np

def collect_training(contour, ct_01):
    batch_size = ct_01.size(0)
    contour = torch.cat([contour[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return contour


def get_gcn_feature(cnn_feature, contour, ind, h, w):
    contour = contour.clone()
    contour[..., 0] = contour[..., 0] / (w / 2.) - 1
    contour[..., 1] = contour[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([contour.size(0), cnn_feature.size(1), contour.size(1)]).to(contour.device)
    for i in range(batch_size):
        contour_ = contour[ind == i].unsqueeze(0).type(torch.float32)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], contour_)[0].permute(1, 0, 2)       # 相邻四个点进行双线性插值
        gcn_feature[ind == i] = feature

    return gcn_feature


def get_adj_mat(n_adj, n_nodes, device):
    a = np.zeros([n_nodes, n_nodes])

    for i in range(n_nodes):
        for j in range(-n_adj // 2, n_adj // 2 + 1):
            if j != 0:
                a[i][(i + j) % n_nodes] = 1
                a[(i + j) % n_nodes][i] = 1

    a = torch.Tensor(a.astype(np.float32))
    return a.to(device)


def get_adj_ind(n_adj, n_nodes, device):
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0])
    ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
    return ind.to(device)


def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    return can_poly


def zoom_poly(poly, scale):
    mean = (poly.min(dim=1, keepdim=True)[0] + poly.max(dim=1, keepdim=True)[0]) * 0.5
    poly = poly - mean
    poly = poly * scale + mean
    return poly

def generate_contour(boxes, num=0):
    h, w = boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]
    b, a = (boxes[:, 3] + boxes[:, 1])/2, (boxes[:, 2] + boxes[:, 0])/2
    alpha = (2 * math.pi) / num     # 角度
    ind = torch.linspace(0, num-1, num).cuda()

    x = a.unsqueeze(0).t() + 0.5 * torch.matmul(w.unsqueeze(0).t(), torch.sin(alpha * ind).unsqueeze(0))
    y = b.unsqueeze(0).t() + 0.5 * torch.matmul(h.unsqueeze(0).t(), torch.cos(alpha * ind).unsqueeze(0))
    point = torch.cat([x.unsqueeze(dim=-1), y.unsqueeze(dim=-1)], dim=-1)
    return point


def init_cirpoint(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}

    init.update({'i_it_ctrs': collect_training(batch['i_it_ctrs'], ct_01)})
    init.update({'c_it_ctrs': collect_training(batch['c_it_ctrs'], ct_01)})
    init.update({'i_gt_ctrs': collect_training(batch['i_gt_ctrs'], ct_01)})
    init.update({'c_gt_ctrs': collect_training(batch['c_gt_ctrs'], ct_01)})

    ct_num = batch['ct_num']
    init.update({'ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init['ind'] = init['ind'].to(ct_01.device)
    return init


def test_cirpoint(dets, points, out_thresh):
    bboxes = dets['bboxes']
    scores = dets['scores']

    ind = scores > out_thresh
    boxes = bboxes[ind]
    ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)

    i_it_ctrs = generate_contour(boxes, num=points)
    i_it_ctrs = i_it_ctrs.cuda()
    c_it_ctrs = img_poly_to_can_poly(i_it_ctrs)
    evolve = {'i_it_ctrs': i_it_ctrs, 'c_it_ctrs': c_it_ctrs, 'ind': ind}
    return evolve
