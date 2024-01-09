import numpy as np


mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.28863828, 0.27408164, 0.27809835],
               dtype=np.float32).reshape(1, 1, 3)
data_rng = np.random.RandomState(123)
eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                   dtype=np.float32)
eig_vec = np.array([
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)

down_ratio = 4
scale = np.array([300, 300])
input_w, input_h = (300, 300)
scale_range = np.arange(0.6, 1.4, 0.1)

voc_input_h, voc_input_w = (512, 512)
# voc_input_h, voc_input_w = (320, 320)
voc_scale_range = np.arange(0.6, 1.4, 0.1)

box_center = False
center_scope = False

init = 'quadrangle'
init_poly_num = 20
poly_num = 128
gt_poly_num = 128
spline_num = 10

adj_num = 4

train_pred_box = False
box_iou = 0.7
confidence = 0.1
train_pred_box_only = True

train_pred_ex = False
train_nearest_gt = True

ro = 4

segm_or_bbox = 'segm'

