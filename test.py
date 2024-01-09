from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import cv2
import json
import copy
import numpy as np
import scipy
import json
import pickle
from lib.opts import opts
from lib.detector_gcn import Detector
import matplotlib.pyplot as plt

image_ext = ['jpg', 'jpeg', 'png', 'bmp']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  detector = Detector(opt)

  if os.path.isdir(opt.demo):
    image_names = []
    ls = os.listdir(opt.demo)
    ls.sort(key=lambda x: int(x.split('.')[0]))
    for file_name in ls:
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in image_ext:
            image_names.append(os.path.join(opt.demo, file_name))
  else:
    image_names = [opt.demo]

  out = None
  out_name = opt.demo[opt.demo.rfind('/') + 1:]
  print('out_name', out_name)
  
  if opt.debug < 5:
    detector.pause = False
  cnt = 0
  results = {}

  while True:
      if cnt < len(image_names):
        img = cv2.imread(image_names[cnt])
      else:
        save_and_exit(opt, out, results, out_name)
      cnt += 1
      ret = detector.run(img, image_names, cnt)

      # log run time
      time_str = 'frame {} |'.format(cnt)
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

      results[cnt-1] = ret['results']

      # save debug image to video
      if opt.output_imgs:
        cv2.imwrite('exp/results/imgs/demo{}.jpg'.format(cnt), ret['generic'])

      # esc to quit and finish saving video
      if cv2.waitKey(1) == 27:
        save_and_exit(opt, out, results, out_name)
        return



def save_and_exit(opt, out=None, results=None, out_name=''):
  if opt.save_results and (results is not None):
    save_dir =  '../results/{}_results.json'.format(opt.input_mode + '_' + out_name)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_dir, 'w'))
  if opt.output_imgs and out is not None:
    out.release()
  sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results


if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
