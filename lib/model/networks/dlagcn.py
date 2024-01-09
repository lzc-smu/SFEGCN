import torch
import torch.nn as nn
from .dla import DLASeg
from .GCN.resgcn import Evolution
from lib.model.decode import generic_decode


class DLAGCN(nn.Module):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLAGCN, self).__init__()

        self.opt = opt
        self.dla = DLASeg(num_layers, heads=heads, head_convs=head_convs, opt=opt)
        self.gcn = Evolution()

    def _sigmoid_output(self, outputs):
        output = outputs[0]
        if 'hm' in output:
            output['hm'] = output['hm'].sigmoid_()
        if 'hm_hp' in output:
            output['hm_hp'] = output['hm_hp'].sigmoid_()
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
            output['dep'] *= self.opt.depth_scale
        return outputs


    def forward(self, img, batch, pre_img=None, pre_hm=None):
        outputs, feat = self.dla(img, pre_img, pre_hm)
        with torch.no_grad():
            outputs = self._sigmoid_output(outputs)
            hm = outputs[0]['hm'].detach().cpu().numpy()
            dets = generic_decode(outputs[0], K=self.opt.K, opt=self.opt)
            outputs[0]['dets'] = dets
        outputs = self.gcn(outputs, feat, batch)
        return outputs
