from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.utils.data
from lib.opts import opts


from lib.model.model import create_model, load_model, save_model
from lib.logger import Logger
from lib.trainer import Trainer
from lib.dataset.pat import Dataset


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  logger = Logger(opt)

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr, eps=1e-4)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up train data...')
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=True,
      num_workers=opt.num_workers, pin_memory=True, drop_last=True)

  print('Setting up validation data...')
  val_loader = torch.utils.data.DataLoader(
    Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1,
    pin_memory=True)

  print('Starting training...')
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
    logger.write('\n')
    if (epoch + 1) % opt.save_point[0] == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
      _, preds = trainer.val(0, val_loader)

    if epoch in opt.lr_step:
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
