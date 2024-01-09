from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import json
import cv2
import os
from collections import defaultdict
import pycocotools.coco as coco
import torch.utils.data as data
from lib.utils.image import get_affine_transform, affine_transform, gaussian_radius, draw_umich_gaussian, color_aug
import copy

from lib.utils.init_contour import init_contour, visualize


class GenericDataset(data.Dataset):
    is_fusion_dataset = False
    default_resolution = None
    num_categories = None
    class_name = None
    cat_ids = None
    max_objs = None
    rest_focal_length = 1200
    num_joints = 17
    flip_idx = []
    edges = []
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                        dtype=np.float32)
    _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    ignore_val = 1
    nuscenes_att_range = {0: [0, 1], 1: [0, 1], 2: [2, 3, 4], 3: [2, 3, 4],
                          4: [2, 3, 4], 5: [5, 6, 7], 6: [5, 6, 7], 7: [5, 6, 7]}

    def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
        super(GenericDataset, self).__init__()
        if opt is not None and split is not None:
            self.split = split
            self.opt = opt
            self._data_rng = np.random.RandomState(123)
            self.init_contour = init_contour
            self.visualize = visualize

        if ann_path is not None and img_dir is not None:
            print('==> initializing {} data from {}, \n images from {} ...'.format(
                split, ann_path, img_dir))
            self.coco = coco.COCO(ann_path)
            self.images = self.coco.getImgIds()

            if opt.tracking:
                if not ('videos' in self.coco.dataset):
                    self.fake_video_data()
                print('Creating video index!')
                self.video_to_images = defaultdict(list)
                for image in self.coco.dataset['images']:
                    self.video_to_images[image['video_id']].append(image)

            self.img_dir = img_dir

    def __getitem__(self, index):
        opt = self.opt
        img, anns, img_info, img_path = self._load_data(index)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
            else np.array([img.shape[1], img.shape[0]], np.float32)
        aug_s, rot, flipped = 1, 0, 0
        if self.split == 'train':
            c, aug_s, rot = self._get_aug_param(c, s, width, height)
            s = s * aug_s

        trans_input = get_affine_transform(
            c, s, rot, [opt.input_w, opt.input_h])
        trans_output = get_affine_transform(
            c, s, rot, [opt.output_w, opt.output_h])
        inp = self._get_input(img, trans_input)
        ret = {'image': inp}
        gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

        if opt.tracking:
            pre_image, pre_anns, frame_dist = self._load_pre_data(
                img_info['video_id'], img_info['frame_id'],
                img_info['sensor_id'] if 'sensor_id' in img_info else 1, self.opt.save_framerate)
            pre_img, pre_hms, pre_cts, multi_slice_ids = [], [], [], []

            for i in range(len(pre_image)):
                if flipped:
                    pre_image[i] = pre_image[i][:, ::-1, :].copy()
                    pre_anns[i] = self._flip_anns(pre_anns[i], width)
                c_pre, aug_s_pre, _ = self._get_aug_param(
                    c, s, width, height, disturb=True)
                s_pre = s * aug_s_pre
                trans_input_pre = get_affine_transform(
                    c_pre, s_pre, rot, [opt.input_w, opt.input_h])
                trans_output_pre = get_affine_transform(
                    c_pre, s_pre, rot, [opt.output_w, opt.output_h])
                pre_img_input = self._get_input(pre_image[i], trans_input_pre)
                pre_hm, pre_ct, multi_slice_id = self._get_pre_dets(
                    pre_anns[i], anns, trans_input_pre, trans_output_pre)
                pre_img.append(pre_img_input)
                if pre_hm is not None:
                    pre_hms.append(pre_hm)
                pre_cts.append(pre_ct)
                multi_slice_ids.append(multi_slice_id)
            pre_imgs = np.zeros(
                (self.opt.vol_slices * 2, pre_img[0].shape[0], pre_img[0].shape[1], pre_img[0].shape[2]), dtype=np.float32)
            pre_imgs[:len(pre_img)] = np.asarray(pre_img)
            ret['pre_img'] = pre_imgs
            pre_hms_ = np.zeros((self.opt.vol_slices * 2, 1, pre_img[0].shape[1], pre_img[0].shape[2]), dtype=np.float32)
            if len(pre_hms) > 0:
                pre_hms_[:len(pre_hms)] = np.asarray(pre_hms)
            if opt.pre_hm:
                ret['pre_hm'] = pre_hms_

        self._init_ret(ret, gt_det)
        calib = self._get_calib(img_info, width, height)
        ret, cls = self.init_contour(ret, anns, flipped, trans_output, self.opt.gcn_hidlayers)

        num_objs = min(len(anns), self.max_objs)
        for k in range(num_objs):
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > self.opt.num_classes or cls_id <= -999:
                continue
            bbox, bbox_amodal = self._get_bbox_output(
                ann['bbox'], trans_output, height, width)
            if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox)
                continue
            self._add_instance(
                ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, pre_anns, trans_output, aug_s,
                calib, pre_cts, multi_slice_ids)

        if self.opt.debug > 0:
            gt_det = self._format_gt_det(gt_det)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
                    'img_path': img_path, 'calib': calib,
                    'flipped': flipped, 'split': self.split}
            ret['meta'] = meta
        else:
            meta = {'split': self.split}
            ret['meta'] = meta
        return ret

    def get_default_calib(self, width, height):
        calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                          [0, self.rest_focal_length, height / 2, 0],
                          [0, 0, 1, 0]])
        return calib

    def _load_image_anns(self, img_id, coco, img_dir):
        img_info = coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        img = cv2.imread(img_path)
        return img, anns, img_info, img_path

    def _load_data(self, index):
        coco = self.coco
        img_dir = self.img_dir
        img_id = self.images[index]
        img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)

        return img, anns, img_info, img_path

    def _load_pre_data(self, video_id, frame_id, sensor_id=1, slice_n=15):
        pre_num = video_id % slice_n
        pos_num = slice_n - 1 - (video_id % slice_n)
        if pre_num > self.opt.vol_slices:
            pre_num = self.opt.vol_slices
        if pos_num > self.opt.vol_slices:
            pos_num = self.opt.vol_slices
        Num = list(range(-pre_num, 0, 1)) + list(range(1, pos_num + 1, 1))
        imgs, anns, frame_dists = [], [], []
        for i in Num:
            img_infos = self.video_to_images[video_id + i]
            if 'train' in self.split:
                img_ids = [(img_info['id'], img_info['frame_id']) for img_info in img_infos]
            else:
                img_ids = [(img_info['id'], img_info['frame_id']) \
                           for img_info in img_infos \
                           if (img_info['frame_id'] - frame_id) == -1 and \
                           (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
                if len(img_ids) == 0:
                    img_ids = [(img_info['id'], img_info['frame_id']) \
                               for img_info in img_infos \
                               if (img_info['frame_id'] - frame_id) == 0 and \
                               (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
            try:
                rand_id = np.random.choice(len(img_ids))
            except:
                print('index: {:.2f}'.format(video_id))
            img_id, pre_frame_id = img_ids[rand_id]
            frame_dist = abs(frame_id - pre_frame_id)
            img, ann, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
            imgs.append(img)
            anns.append(ann)
            frame_dists.append(frame_dist)
        return imgs, anns, frame_dists

    def _get_pre_dets(self, anns, inp, trans_input, trans_output):
        hm_h, hm_w = self.opt.input_h, self.opt.input_w
        down_ratio = self.opt.down_ratio
        trans = trans_input
        if anns[0]['image_id'] < inp[0]['image_id']:
            reutrn_hm = self.opt.pre_hm
        else:
            reutrn_hm = False
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
        pre_cts, multi_slice_ids = [], []
        for ann in anns:
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > self.opt.num_classes or cls_id <= -99 or \
                    ('iscrowd' in ann and ann['iscrowd'] > 0):
                continue
            bbox = self._coco_box_to_bbox(ann['bbox'])
            bbox[:2] = affine_transform(bbox[:2], trans)
            bbox[2:] = affine_transform(bbox[2:], trans)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            max_rad = 1
            if (h > 0 and w > 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                max_rad = max(max_rad, radius)
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct0 = ct.copy()
                conf = 1

                ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
                ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
                conf = 1 if np.random.random() > self.opt.lost_disturb else 0

                ct_int = ct.astype(np.int32)
                if conf == 0:
                    pre_cts.append(ct / down_ratio)
                else:
                    pre_cts.append(ct0 / down_ratio)

                multi_slice_ids.append(ann['track_id'] if 'track_id' in ann else -1)
                if reutrn_hm:
                    draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

                if np.random.random() < self.opt.fp_disturb and reutrn_hm:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h
                    ct2_int = ct2.astype(np.int32)
                    draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

        return pre_hm, pre_cts, multi_slice_ids

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_aug_param(self, c, s, width, height, disturb=False):
        if (not self.opt.not_rand_crop) and not disturb:
            aug_s = np.random.choice(np.arange(0.8, 1.4, 0.1))
            w_border = self._get_border(128, width)
            h_border = self._get_border(128, height)
            c[0] = np.random.randint(low=w_border, high=width - w_border)
            c[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            sf = self.opt.scale
            cf = self.opt.shift
            if type(s) == float:
                # s = [s, s]
                s = np.float64(s)
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.opt.aug_rot:
            rf = self.opt.rotate
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            rot = 0

        return c, aug_s, rot

    def _flip_anns(self, anns, width):
        for k in range(len(anns)):
            bbox = anns[k]['bbox']
            anns[k]['bbox'] = [
                width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

            if 'hps' in self.opt.heads and 'keypoints' in anns[k]:
                keypoints = np.array(anns[k]['keypoints'], dtype=np.float32).reshape(
                    self.num_joints, 3)
                keypoints[:, 0] = width - keypoints[:, 0] - 1
                for e in self.flip_idx:
                    keypoints[e[0]], keypoints[e[1]] = \
                        keypoints[e[1]].copy(), keypoints[e[0]].copy()
                anns[k]['keypoints'] = keypoints.reshape(-1).tolist()
        return anns

    def _get_input(self, img, trans_input):
        inp = cv2.warpAffine(img, trans_input,
                            (self.opt.input_w, self.opt.input_h),
                            flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def _init_ret(self, ret, gt_det):
        max_objs = self.max_objs * self.opt.dense_reg
        ret['hm'] = np.zeros(
            (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
            np.float32)
        ret['ind'] = np.zeros((max_objs), dtype=np.int64)
        ret['cat'] = np.zeros((max_objs), dtype=np.int64)
        ret['mask'] = np.zeros((max_objs), dtype=np.float32)

        regression_head_dims = {
            'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
            'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2,
            'dep': 1, 'dim': 3, 'amodel_offset': 2}

        for head in regression_head_dims:
            if head in self.opt.heads:
                ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                ret[head + '_mask'] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                gt_det[head] = []

    def _get_calib(self, img_info, width, height):
        if 'calib' in img_info:
            calib = np.array(img_info['calib'], dtype=np.float32)
        else:
            calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                              [0, self.rest_focal_length, height / 2, 0],
                              [0, 0, 1, 0]])
        return calib

    def _ignore_region(self, region, ignore_val=1):
        np.maximum(region, ignore_val, out=region)

    def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
        # mask out crowd region, only rectangular mask is supported
        if cls_id == 0:  # ignore all classes
            self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])
        else:
            # mask out one specific class
            self._ignore_region(ret['hm'][abs(cls_id) - 1,
                                int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])
        if ('hm_hp' in ret) and cls_id <= 1:
            self._ignore_region(ret['hm_hp'][:, int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_bbox_output(self, bbox, trans_output, height, width):
        bbox = self._coco_box_to_bbox(bbox).copy()

        rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                         [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
        for t in range(4):
            rect[t] = affine_transform(rect[t], trans_output)
        bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
        bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

        bbox_amodal = copy.deepcopy(bbox)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        return bbox, bbox_amodal

    def info_isin(self, number, k, lists):
        for ls in lists:
            if number - k -1 in ls:
                return True
        return False

    def info_search(self, number, k, lists):
        n = number - k - 1
        for i in range(len(lists)):
            if n in lists[i]:
                return i
        return 0

    def select_ct_tracking(self, pre_cts, ct_int, pre_anns, id, track_ids):
        pre_id = [ann['category_id'] for ann in pre_anns]
        ct_id = [x for x, y in list(enumerate(pre_id)) if y == id]
        for i in range(len(ct_id)):
            track_id = pre_anns[ct_id[i]]['track_id']
            if self.info_isin(track_id, -1, track_ids):
                info = self.info_search(track_id, -1, track_ids)
                ct_id[i] = track_ids[info].index(track_id)
            else:
                ct_id.remove(ct_id[i])

        track = None
        if len(ct_id) != 0:
            if len(ct_id) > 1:
                tracks = [pre_cts[c] - ct_int for c in ct_id]
                diff = [(tracks[i][0]**2 + tracks[i][1]**2)**0.5 for i in range(len(tracks))]
                track = tracks[diff.index(min(diff))]
            else:
                track = pre_cts[ct_id[0]] - ct_int
        return track

    def _add_instance(
            self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, pre_anns, trans_output,
            aug_s, calib, pre_cts=None, track_ids=None):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h <= 0 or w <= 0:
            return
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        ret['cat'][k] = cls_id - 1
        ret['mask'][k] = 1
        if 'wh' in ret:
            ret['wh'][k] = 1. * w, 1. * h
            ret['wh_mask'][k] = 1
        ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
        ret['reg'][k] = ct - ct_int
        ret['reg_mask'][k] = 1
        draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

        gt_det['bboxes'].append(
            np.array([ct[0] - w / 2, ct[1] - h / 2,
                      ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
        gt_det['scores'].append(1)
        gt_det['clses'].append(cls_id - 1)
        gt_det['cts'].append(ct)

        if 'tracking' in self.opt.heads:
            # if self.info_isin(ann['track_id'], k, track_ids):
            #     info = self.info_search(ann['track_id'], k, track_ids)
            #     tracking = self.select_ct_tracking(pre_cts[info], ct_int, pre_anns[info], ann['category_id'], track_ids)
            #     if tracking is not None:
            #         ret['tracking_mask'][k] = 1
            #         ret['tracking'][k] = tracking
            #         gt_det['tracking'].append(ret['tracking'][k])
            # else:
            #     if k < len(pre_cts[0]):
            #         pre_ct = pre_cts[0][k]
            #         ret['tracking_mask'][k] = 1
            #         ret['tracking'][k] = pre_ct - ct_int
            #         gt_det['tracking'].append(ret['tracking'][k])
            tn = 0
            track = np.array([0, 0], dtype = np.float64)
            for i in range(len(pre_anns)):
                tracking = self.select_ct_tracking(pre_cts[i], ct_int, pre_anns[i], ann['category_id'], track_ids)
                if tracking is not None:
                    track += tracking
                    tn += 1
            if tn > 0:
                ret['tracking_mask'][k] = 1
                ret['tracking'][k] = track/tn
                gt_det['tracking'].append(ret['tracking'][k])
            else:
                if k < len(pre_cts[0]):
                    pre_ct = pre_cts[0][k]
                    ret['tracking_mask'][k] = 1
                    ret['tracking'][k] = pre_ct - ct_int
                    gt_det['tracking'].append(ret['tracking'][k])


    def _format_gt_det(self, gt_det):
        if (len(gt_det['scores']) == 0):
            gt_det = {'bboxes': np.array([[0, 0, 1, 1]], dtype=np.float32),
                      'scores': np.array([1], dtype=np.float32),
                      'clses': np.array([0], dtype=np.float32),
                      'cts': np.array([[0, 0]], dtype=np.float32),
                      'pre_cts': np.array([[0, 0]], dtype=np.float32),
                      'tracking': np.array([[0, 0]], dtype=np.float32),
                      'bboxes_amodal': np.array([[0, 0]], dtype=np.float32),
                      'hps': np.zeros((1, 17, 2), dtype=np.float32), }
        gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
        return gt_det

    def fake_video_data(self):
        self.coco.dataset['videos'] = []
        for i in range(len(self.coco.dataset['images'])):
            img_id = self.coco.dataset['images'][i]['id']
            self.coco.dataset['images'][i]['video_id'] = img_id
            self.coco.dataset['images'][i]['frame_id'] = 1
            self.coco.dataset['videos'].append({'id': img_id})

        if not ('annotations' in self.coco.dataset):
            return

        for i in range(len(self.coco.dataset['annotations'])):
            self.coco.dataset['annotations'][i]['track_id'] = i + 1
