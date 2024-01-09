import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def init_contour(ret, ann, flipped, trans, point):
    contour_num = point
    num = 16
    contour = [[np.array(s).reshape(-1, 2) for s in obj['segmentation']] for obj in ann]
    contour = transform_data(contour, flipped, 300, trans, (128, 128))
    contour = get_valid_polys(contour)

    i_it_ctrs = np.zeros([num, contour_num, 2])
    c_it_ctrs = np.zeros([num, contour_num, 2])
    i_gt_ctrs = np.zeros([num, contour_num, 2])
    c_gt_ctrs = np.zeros([num, contour_num, 2])
    ct_01 = np.zeros([num], dtype=bool)
    cls = []
    ct_num = 0
    for i in range(len(contour)):
        if len(contour[i]) != 0:
            img_init_contour = init_uniform_cirsample(contour[i][0], contour_num)
            can_init_contour = img_poly_to_can_poly(img_init_contour)

            img_gt_contour= uniformsample(contour[i][0], len(contour[i][0]) * contour_num)
            tt_idx = np.argmin(np.power(img_gt_contour - img_init_contour[0], 2).sum(axis=1))
            img_gt_contour = np.roll(img_gt_contour, -tt_idx, axis=0)[::len(contour[i][0])]
            can_gt_contour = img_poly_to_can_poly(img_gt_contour)

            i_it_ctrs[ct_num] = img_init_contour
            c_it_ctrs[ct_num] = can_init_contour
            i_gt_ctrs[ct_num] = img_gt_contour
            c_gt_ctrs[ct_num] = can_gt_contour
            ct_01[ct_num] = 1
            ct_num += 1
            cls.append(ann[i]['category_id'])

    ret['i_it_ctrs'] = i_it_ctrs
    ret['c_it_ctrs'] = c_it_ctrs
    ret['i_gt_ctrs'] = i_gt_ctrs
    ret['c_gt_ctrs'] = c_gt_ctrs
    ret['ct_01'] = ct_01
    ret['ct_num'] = ct_num
    return ret, np.asarray(cls)

def transform_data(contours, flipped, width, trans_output, inp_out_hw):
    output_h, output_w = inp_out_hw[:]
    instance_ctrs_ = []
    for instance in contours:
        ctrs = [ctr.reshape(-1, 2) for ctr in instance]
        if flipped:
            ctrs_ = []
            for ctr in ctrs:
                ctr[:, 0] = width - np.array(ctr[:, 0]) - 1
                ctrs_.append(ctr.copy())
            ctrs = ctrs_
        output = transform_polys(ctrs, trans_output, output_h, output_w)
        instance_ctrs_.append(output)
    return instance_ctrs_

def get_valid_polys(instance_polys):
    instance_polys_ = []
    for instance in instance_polys:
        polys = filter_tiny_polys(instance)
        polys = get_cw_polys(polys)
        polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]

        instance_polys_.append(polys)
    return instance_polys_

def affine_transform(pt, t):
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt

def transform_polys(polys, trans_output, output_h, output_w):
    new_polys = []
    for poly in polys:
        poly = affine_transform(poly, trans_output)
        poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
        poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
        if len(poly) == 0:
            continue
        if len(np.unique(poly, axis=0)) <= 2:
            continue
        new_polys.append(poly)
    return new_polys

def handle_break_point(poly, axis, number, outside_border):
    if len(poly) == 0:
        return []

    if len(poly[outside_border(poly[:, axis], number)]) == len(poly):
        return []

    break_points = np.argwhere(
        outside_border(poly[:-1, axis], number) != outside_border(poly[1:, axis], number)).ravel()
    if len(break_points) == 0:
        return poly

    new_poly = []
    if not outside_border(poly[break_points[0], axis], number):
        new_poly.append(poly[:break_points[0]])

    for i in range(len(break_points)):
        current_poly = poly[break_points[i]]
        next_poly = poly[break_points[i] + 1]
        mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (next_poly[axis] - current_poly[axis])

        if outside_border(poly[break_points[i], axis], number):
            if mid_poly[axis] != next_poly[axis]:
                new_poly.append([mid_poly])
            next_point = len(poly) if i == (len(break_points) - 1) else break_points[i + 1]
            new_poly.append(poly[break_points[i] + 1:next_point])
        else:
            new_poly.append([poly[break_points[i]]])
            if mid_poly[axis] != current_poly[axis]:
                new_poly.append([mid_poly])

    if outside_border(poly[-1, axis], number) != outside_border(poly[0, axis], number):
        current_poly = poly[-1]
        next_poly = poly[0]
        mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (next_poly[axis] - current_poly[axis])
        new_poly.append([mid_poly])

    return np.concatenate(new_poly)

def filter_tiny_polys(polys):
    return [poly for poly in polys if Polygon(poly).area > 5]

def get_cw_polys(polys):
    return [poly[::-1] if Polygon(poly).exterior.is_ccw else poly for poly in polys]

def img_poly_to_can_poly(img_poly):
    x_min, y_min = np.min(img_poly, axis=0)
    can_poly = img_poly - np.array([x_min, y_min])
    return can_poly

def init_uniform_cirsample(poly, num):
    x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
    x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
    h, w = y_max - y_min + 1, x_max - x_min + 1
    a, b = (x_min + x_max) / 2, (y_min + y_max) / 2

    alpha = np.linspace(0, 2*math.pi*(num-1)/num, num)
    x = np.reshape(a + 0.5 * w * np.sin(alpha), (-1, 1))
    y = np.reshape(b + 0.5 * h * np.cos(alpha), (-1, 1))
    point = np.concatenate((x, y), axis=1)
    return point

def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp

def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]

def visualize(img, data):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = bgr_to_rgb(img.transpose(1, 2, 0))
    plt.imshow(img)
    for poly in data:
        poly = poly * 4
        poly = np.append(poly, [poly[0]], axis=0)
        plt.plot(poly[:, 0], poly[:, 1])
        plt.scatter(poly[0, 0], poly[0, 1], edgecolors='w')

    # plt.savefig(os.path.join('demo_images/' + 'fig{}.png'.format(2)))
    plt.show()
