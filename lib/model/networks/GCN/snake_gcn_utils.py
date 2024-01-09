import torch
import math
import random
import numpy as np
from . import snake_decode, snake_config
# from lib.csrc.extreme_utils import _ext as extreme_utils


def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return poly


def prepare_training_init(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})

    return init


def prepare_testing_init(box, score):
    i_it_4pys = snake_decode.get_init(box)
    i_it_4pys = uniform_upsample(i_it_4pys, snake_config.init_poly_num)
    c_it_4pys = img_poly_to_can_poly(i_it_4pys)

    ind = score > 0.1
    i_it_4pys = i_it_4pys[ind]
    c_it_4pys = c_it_4pys[ind]
    ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)
    init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'ind': ind}

    return init


def get_box_match_ind(pred_box, score, gt_poly):
    if gt_poly.size(0) == 0:
        return [], []

    gt_box = torch.cat([torch.min(gt_poly, dim=1)[0], torch.max(gt_poly, dim=1)[0]], dim=1)
    iou_matrix = data_utils.box_iou(pred_box, gt_box)
    iou, gt_ind = iou_matrix.max(dim=1)
    box_ind = ((iou > snake_config.box_iou) * (score > snake_config.confidence)).nonzero().view(-1)
    gt_ind = gt_ind[box_ind]

    ind = np.unique(gt_ind.detach().cpu().numpy(), return_index=True)[1]
    box_ind = box_ind[ind]
    gt_ind = gt_ind[ind]

    return box_ind, gt_ind


def prepare_training_box(ret, batch, init):
    box = ret['detection'][..., :4]
    score = ret['detection'][..., 4]
    batch_size = box.size(0)
    i_gt_4py = batch['i_gt_4py']
    ct_01 = batch['ct_01'].byte()
    ind = [get_box_match_ind(box[i], score[i], i_gt_4py[i][ct_01[i]]) for i in range(batch_size)]
    box_ind = [ind_[0] for ind_ in ind]
    gt_ind = [ind_[1] for ind_ in ind]

    i_it_4py = torch.cat([snake_decode.get_init(box[i][box_ind[i]][None]) for i in range(batch_size)], dim=1)
    if i_it_4py.size(1) == 0:
        return

    i_it_4py = uniform_upsample(i_it_4py, snake_config.init_poly_num)[0]
    c_it_4py = img_poly_to_can_poly(i_it_4py)
    i_gt_4py = torch.cat([batch['i_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    c_gt_4py = torch.cat([batch['c_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_4py = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'i_gt_4py': i_gt_4py, 'c_gt_4py': c_gt_4py}

    i_it_py = snake_decode.get_octagon(i_gt_4py[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    i_gt_py = torch.cat([batch['i_gt_py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_py = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    ind = torch.cat([torch.full([len(gt_ind[i])], i) for i in range(batch_size)], dim=0)

    if snake_config.train_pred_box_only:
        for k, v in init_4py.items():
            init[k] = v
        for k, v in init_py.items():
            init[k] = v
        init['4py_ind'] = ind
        init['py_ind'] = ind
    else:
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_4py.items()})
        init.update({'4py_ind': torch.cat([init['4py_ind'], ind], dim=0)})
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_py.items()})
        init.update({'py_ind': torch.cat([init['py_ind'], ind], dim=0)})


def prepare_training(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    init.update({'i_it_py': collect_training(batch['i_it_py'], ct_01)})
    init.update({'c_it_py': collect_training(batch['c_it_py'], ct_01)})
    init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})
    init.update({'c_gt_py': collect_training(batch['c_gt_py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'4py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['4py_ind']})

    # if snake_config.train_pred_box:
    #     prepare_training_box(ret, batch, init)

    init['4py_ind'] = init['4py_ind'].to(ct_01.device)
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init


# def prepare_training_evolve(ex, init):
#     if not snake_config.train_pred_ex:
#         evolve = {'i_it_py': init['i_it_py'], 'c_it_py': init['c_it_py'], 'i_gt_py': init['i_gt_py']}
#         return evolve
#
#     i_gt_py = init['i_gt_py']
#
#     if snake_config.train_nearest_gt:
#         shift = -(ex[:, :1] - i_gt_py).pow(2).sum(2).argmin(1)
#         i_gt_py = extreme_utils.roll_array(i_gt_py, shift)
#
#     i_it_py = snake_decode.get_octagon(ex[None])
#     i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
#     c_it_py = img_poly_to_can_poly(i_it_py)
#     evolve = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}
#
#     return evolve


def prepare_testing_evolve(ex):
    # ex：torch.Size（[240，4，2]）获取八边形后并进行了一次上采样
    if len(ex) == 0:
        i_it_pys = torch.zeros([0, snake_config.poly_num, 2]).to(ex)
        c_it_pys = torch.zeros_like(i_it_pys)
    else:
        i_it_pys = snake_decode.get_octagon(ex[None])       # 生成八边形
        i_it_pys = uniform_upsample(i_it_pys, snake_config.poly_num)[0]     # 采样128个点
        c_it_pys = img_poly_to_can_poly(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve


def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1      # 归一化到（-1，1）
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0).type(torch.float32)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)       # 相邻四个点进行双线性插值
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


def get_pconv_ind(n_adj, n_nodes, device):
    n_outer_nodes = snake_config.poly_num
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1)])
    outer_ind = (torch.arange(n_outer_nodes)[:, None] + ind[None]) % n_outer_nodes
    inner_ind = outer_ind + n_outer_nodes
    ind = torch.cat([outer_ind, inner_ind], dim=1)
    return ind


def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    # x_max = torch.max(img_poly[..., 0], dim=-1)[0]
    # y_max = torch.max(img_poly[..., 1], dim=-1)[0]
    # h, w = y_max - y_min + 1, x_max - x_min + 1
    # long_side = torch.max(h, w)
    # can_poly = can_poly / long_side[..., None, None]
    return can_poly


def img_poly_to_can_poly_3d(img_poly):
    can_polys = []
    for i in range(img_poly.__len__()):
        # if len(img_poly[i]) == 0:
        #     return [torch.zeros_like(img_poly[i])]
        x_min = torch.min(img_poly[i][..., 1], dim=-1)[0]
        y_min = torch.min(img_poly[i][..., 2], dim=-1)[0]
        can_poly = img_poly[i].clone()
        can_poly[..., 1] = can_poly[..., 1] - x_min[..., None]
        can_poly[..., 2] = can_poly[..., 2] - y_min[..., None]
        can_polys.append(can_poly)
    return can_polys


def uniform_upsample(poly, p_num):
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation
    next_poly = torch.roll(poly, -1, 2)
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()
    edge_num = torch.round(edge_len * p_num / torch.sum(edge_len, dim=2)[..., None]).long()
    edge_num = torch.clamp(edge_num, min=1)
    edge_num_sum = torch.sum(edge_num, dim=2)
    edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
    extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
    poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly = poly1 * (1 - weight) + poly2 * weight

    return poly


def zoom_poly(poly, scale):
    mean = (poly.min(dim=1, keepdim=True)[0] + poly.max(dim=1, keepdim=True)[0]) * 0.5
    poly = poly - mean
    poly = poly * scale + mean
    return poly

def generate_contour(boxes, num=0):
    # h, w = boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]
    # a, b = (boxes[:, 3] + boxes[:, 1])/2, (boxes[:, 2] + boxes[:, 0])/2     # 圆心
    h, w = boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]
    b, a = (boxes[:, 3] + boxes[:, 1])/2, (boxes[:, 2] + boxes[:, 0])/2
    alpha = (2 * math.pi) / num     # 角度
    ind = torch.linspace(0, num-1, num).cuda()

    x = a.unsqueeze(0).t() + 0.5 * torch.matmul(w.unsqueeze(0).t(), torch.sin(alpha * ind).unsqueeze(0))
    y = b.unsqueeze(0).t() + 0.5 * torch.matmul(h.unsqueeze(0).t(), torch.cos(alpha * ind).unsqueeze(0))
    point = torch.cat([x.unsqueeze(dim=-1), y.unsqueeze(dim=-1)], dim=-1)
    return point


def uniform_cirsample(feat, init_poly=None, num=0):
    # h, w = feat.size(2), feat.size(3)
    # a, b = h/2, w/2     # 圆心
    # if init_poly is not None:
    #     ct_vect = init_poly[..., 0, :] - torch.tensor([a, b]).cuda()
    #     fp_r = (ct_vect).pow(2).sum().sqrt()
    #     fp_alpha = torch.acos(ct_vect[..., 1]/fp_r)
    #     if ct_vect[..., 0] < 0:
    #         fp_alpha = 2 * math.pi - fp_alpha
    # else:
    #     fp_alpha = 0
    # alpha = (2 * math.pi) / num     # 角度
    # r = 0.7 * min(a, b)    # 半径
    # ind = torch.linspace(0, num-1, num).cuda()

    # x = a + r * torch.sin(alpha * ind + fp_alpha)
    # y = b + r * torch.cos(alpha * ind + fp_alpha)
    # point = torch.cat([x.unsqueeze(dim=-1), y.unsqueeze(dim=-1)], dim=-1)
    # return point

    h, w = feat.size(2), feat.size(3)
    if init_poly is not None:
        h_min, w_min = torch.min(init_poly[..., 0], dim=2).values, torch.min(init_poly[..., 1], dim=2).values
        h_max, w_max = torch.max(init_poly[..., 0], dim=2).values, torch.max(init_poly[..., 1], dim=2).values
        a, b = (h_min + h_max)/2, (w_min + w_max)/2
        # r = random.uniform(max(h_max-a, a-h_min, w_max-b, b-w_min), min((h - a), (w - b), a, b))
        r = torch.min(torch.cat([(h - a), (w - b), a, b], dim=1), dim=1).values.unsqueeze(dim=1)
        alpha = torch.tensor([]).cuda()
        for i in range(num):
            ct_vect = init_poly[..., i, :].squeeze(dim=1) - torch.cat([a, b],dim=1)
            fp_r = (ct_vect).pow(2).sum(dim=-1).sqrt()
            fp_alpha = torch.acos(ct_vect[..., 1]/fp_r)
            for j in range(len(fp_alpha[:, ...])):
                fp_alpha[j] = 2 * math.pi - fp_alpha[j] if ct_vect[j, 0] < 0 else fp_alpha[j]
            alpha = torch.cat([alpha, fp_alpha.unsqueeze(dim=-1)], dim=-1)
        x = a + r * torch.sin(alpha)
        y = b + r * torch.cos(alpha)
        point = torch.cat([x.unsqueeze(dim=-1), y.unsqueeze(dim=-1)], dim=-1)
    else:
        a, b = h/2, w/2
        alpha = torch.linspace(0, 2*math.pi*(num-1)/num, num).cuda()
        r = 0.6 * min(a, b)

        x = a + r * torch.sin(alpha)
        y = b + r * torch.cos(alpha)
        point = torch.cat([x.unsqueeze(dim=-1), y.unsqueeze(dim=-1)], dim=-1)
    point[..., 0] = torch.clamp(point[..., 0], min=0, max=w - 1)
    point[..., 1] = torch.clamp(point[..., 1], min=0, max=h - 1)
    return point




def init_cirpoint(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}

    init.update({'i_it_py': collect_training(batch['i_it_py'], ct_01)})
    init.update({'c_it_py': collect_training(batch['c_it_py'], ct_01)})
    init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})
    init.update({'c_gt_py': collect_training(batch['c_gt_py'], ct_01)})
    init.update({'pre_gt_py': collect_training(batch['pre_py'], ct_01)})

    # i_it_pys = uniform_cirsample(feat, batch['i_gt_py'], snake_config.poly_num)
    # c_it_pys = img_poly_to_can_poly(i_it_pys)
    # init.update({'i_it_py': i_it_pys, 'c_it_py': c_it_pys})

    ct_num = batch['ct_num']
    init.update({'4py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['4py_ind']})

    # if snake_config.train_pred_box:
    #     prepare_training_box(ret, batch, init)

    init['4py_ind'] = init['4py_ind'].to(ct_01.device)
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init


def test_cirpoint(dets, points, out_thresh):
    bboxes = dets['bboxes']
    scores = dets['scores']

    ind = scores > out_thresh
    boxes = bboxes[ind]
    ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)

    i_it_pys = generate_contour(boxes, num=points)
    i_it_pys = i_it_pys.cuda()
    c_it_pys = img_poly_to_can_poly(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'ind': ind}
    return evolve


def init_uniform_cirpoint(init, num=snake_config.poly_num):
    i_it_pys = torch.zeros([init['i_it_4py'].shape[0], num, 2])
    for i in range(init['i_it_4py'].shape[0]):
        poly = init['i_it_4py'][i, ...]
        x_min, y_min = torch.min(poly[:, 0]), torch.min(poly[:, 1])
        x_max, y_max = torch.max(poly[:, 0]), torch.max(poly[:, 1])
        h, w = y_max - y_min + 1, x_max - x_min + 1
        a, b = (x_min + x_max) / 2, (y_min + y_max) / 2

        alpha = torch.linspace(0, 2*math.pi*(num-1)/num, num).cuda()
        x = torch.reshape(a + 0.5 * w * torch.sin(alpha), (-1, 1))
        y = torch.reshape(b + 0.5 * h * torch.cos(alpha), (-1, 1))
        i_it_pys[i, ...] = torch.cat([x, y], dim=-1)
    i_it_pys = i_it_pys.cuda()
    c_it_pys = img_poly_to_can_poly(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve


def init_batchpoint(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}

    init.update({'i_it_py': batch['i_it_py']})
    init.update({'c_it_py': batch['c_it_py']})
    init.update({'i_gt_py': batch['i_gt_py']})
    init.update({'c_gt_py': batch['c_gt_py']})

    ct_num = batch['meta']['ct_num']
    init.update({'4py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['4py_ind']})

    if snake_config.train_pred_box:
        prepare_training_box(ret, batch, init)

    init['4py_ind'] = init['4py_ind'].to(ct_01.device)
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init


def get_3d_gcn_feature(cnn_feature, img_polys, ind, d, h, w):
    gcn_features = []
    for img_poly in img_polys:
        img_poly = img_poly.clone()
        img_poly[..., -3] = img_poly[..., -3] / (d / 2.) - 1      # 归一化到（-1，1）
        img_poly[..., -2] = img_poly[..., -2] / (h / 2.) - 1
        img_poly[..., -1] = img_poly[..., -1] / (w / 2.) - 1
        # ploys = torch.cat([img_poly[..., 1:], img_poly[..., [0]]], dim=3)

        batch_size = cnn_feature.size(0)
        gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)

        for i in range(batch_size):
            poly = img_poly.unsqueeze(0)
            for j in range(poly.size(1)):
                feature = torch.nn.functional.grid_sample(cnn_feature[i:i + 1, :, j, :, :], poly[:, j:j + 1, :, 1:])[0].permute(1, 0, 2)
                gcn_feature[j:j+1, ...] = feature
        gcn_features.append(gcn_feature)
    return gcn_features

def score_threshold(detection):
    num = torch.unique(detection[..., 5])
    detect_class = [[] for i in range(len(num))]
    for i in range(detection.shape[0]):
        detect_sl = detection[i, ...]
        detect_th = detect_sl[detect_sl[..., 4] > snake_config.ct_score]
        sl = torch.ones(detect_th.shape[0]) * i
        detect_th = torch.cat((detect_th, sl.unsqueeze(1)), 1)
        for j in range(detect_th.shape[0]):
            detect_class[int(detect_th[j, 5])].append(detect_th[j])
    detect_class = [x for x in detect_class if x != []]
    class_ = [torch.stack(c) for c in detect_class]
    return class_

def uniform_cirsample_3d(feat, detection, num=0):
    polys = []
    for i in range(len(detection)):
        detection_ = detection[i]
        slice = torch.unique(detection_[..., 6])
        d = int(slice.max().item() - slice.min().item() + 1)
        poly = torch.zeros([d, num, 3]).cuda()
        for j in range(int(slice.min().item()), int(slice.max().item())+1):
            sl_filt = detection_[detection_[:, 6] == j]
            if sl_filt.numel():
                # ind = sl_filt[..., 4].argmax(dim=0).item()
                # x_min, y_min, x_max, y_max = sl_filt[ind, 0], sl_filt[ind, 1], sl_filt[ind, 2], sl_filt[ind, 3]
                ind = detection_[..., 4].argmax(dim=0).item()
                x_min, y_min, x_max, y_max = detection_[ind, 0], detection_[ind, 1], detection_[ind, 2], detection_[ind, 3]
                a, b = (y_min + y_max) / 2, (x_min + x_max) / 2
                alpha = torch.linspace(0, 2*math.pi*(num-1)/num, num).cuda()
                rw, rh = abs(a-y_min), abs(b-x_min)

                x = a + 0.5 * rw * torch.sin(alpha)
                y = b + 0.5 * rh * torch.cos(alpha)
                point = torch.cat([x.unsqueeze(dim=-1), y.unsqueeze(dim=-1)], dim=-1)
                poly[j - int(slice.min().item()), ..., 1:] = point
                poly[j - int(slice.min().item()), ..., 0] = j
            else:
                poly[j - int(slice.min().item()), ..., 1:] = poly[j - int(slice.min().item()) - 1, ..., 1:]
                poly[j - int(slice.min().item()), ..., 0] = j
        polys.append(poly)
    return polys


# def test_cirpoint_3d(feat, detection):
#     i_it_pys = uniform_cirsample_3d(feat, detection, num=snake_config.poly_num)
#     i_it_pys = [it.cuda() for it in i_it_pys]
#     c_it_pys = img_poly_to_can_poly_3d(i_it_pys)
#     evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
#     return evolve

def test_cirpoint_3d(feat, contour):
    i_it_pys = [it.cuda() for it in contour]
    c_it_pys = img_poly_to_can_poly_3d(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve


def uniform_cirsample_ct_3d(output, feat, num=0):
    d, h, w = feat.size(-3), feat.size(-2), feat.size(-1)
    poly = torch.zeros([d, num, 3]).cuda()
    ct_hm = output['ct_hm'].squeeze(dim=0).unsqueeze(dim=1).cpu().numpy()
    m = np.argwhere(ct_hm == ct_hm.max())
    b, a =  m[0, 2:]
    alpha = torch.linspace(0, 2*math.pi*(num-1)/num, num).cuda()
    r = 0.5 * min(a, b, h-a, w-b)

    x = a + r * torch.sin(alpha)
    y = b + r * torch.cos(alpha)
    point = torch.cat([x.unsqueeze(dim=-1), y.unsqueeze(dim=-1)], dim=-1)
    for i in range(d):
        poly[i, ..., 1:] = point
        poly[i, ..., 0] = i
    return poly


def test_ct_cirpoint_3d(box, feat):
    i_it_pys = uniform_cirsample_ct_3d(box, feat, num=snake_config.poly_num).unsqueeze(dim=0)
    i_it_pys = i_it_pys.cuda()
    c_it_pys = img_poly_to_can_poly_3d(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve
