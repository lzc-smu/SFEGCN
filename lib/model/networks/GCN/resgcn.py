import torch.nn as nn
import torch
from .graphSAGE import GCN, construct_graph
from . import gcn_utils


class ResGCN(nn.Module):
    def __init__(self, opt):
        super(ResGCN, self).__init__()

        self.opt = opt
        self.hidlayers = self.opt.gcn_hidlayers
        self.fuse = nn.Conv1d(self.hidlayers, 64, 1)
        self.init_gcn = GCN(state_dim=self.hidlayers, feature_dim=64+2)
        self.resgcn = GCN(state_dim=self.hidlayers, feature_dim=64+2)
        self.iter = 3
        self.ro = 4
        for i in range(self.iter):
            resgcn = GCN(state_dim=self.hidlayers, feature_dim=64+2)
            self.__setattr__('resgcn'+str(i), resgcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_cirpoint(self, output, batch):
        init = gcn_utils.init_cirpoint(output, batch)
        output.update({'i_it_ctrs': init['i_it_ctrs'], 'i_gt_ctrs': init['i_gt_ctrs']})
        return init

    def test_cirpoint(self, output):
        evolve = gcn_utils.test_cirpoint(output['dets'], self.hidlayers, self.opt.out_thresh)
        output.update({'i_it_ctrs': evolve['i_it_ctrs']})
        return evolve

    def evolve_contour(self, model, cnn_feature, i_it_contour, c_it_contour, ind):
        if len(i_it_contour) == 0:
            return torch.zeros_like(i_it_contour)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = gcn_utils.get_gcn_feature(cnn_feature, i_it_contour, ind, h, w)
        c_it_contour = c_it_contour.type(torch.float32) * self.ro
        init_input = torch.cat([init_feature, c_it_contour.permute(0, 2, 1)], dim=1)
        graph = construct_graph(init_input)
        i_poly = i_it_contour * self.ro + model(init_input, graph).permute(0, 2, 1)
        return i_poly

    def forward(self, output, cnn_feature, batch=None):
        ret = output[0]

        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.init_cirpoint(output[0], batch)

            pred  = self.evolve_contour(self.resgcn, cnn_feature, init['i_it_ctrs'], init['c_it_ctrs'], init['ind'])
            preds = [pred]
            for i in range(self.iter):
                pred = pred / self.ro
                c_pred = gcn_utils.img_poly_to_can_poly(pred)
                resgcn = self.__getattr__('resgcn'+str(i))
                pred = self.evolve_contour(resgcn, cnn_feature, pred, c_pred, init['ind'])
                preds.append(pred)
            ret.update({'preds': preds, 'i_gt_ctrs': output[0]['i_gt_ctrs'] * self.ro})

        if batch is None:
            with torch.no_grad():
                init_contour = self.test_cirpoint(output[0])
                ind = init_contour['ind']
                pred = self.evolve_contour(self.resgcn, cnn_feature, init_contour['i_it_ctrs'], init_contour['c_it_ctrs'], ind)
                for i in range(self.iter):
                    pred = pred / self.ro
                    c_pred = gcn_utils.img_poly_to_can_poly(pred)
                    resgcn = self.__getattr__('resgcn'+str(i))
                    pred = self.evolve_contour(resgcn, cnn_feature, pred, c_pred, ind)
                    preds = [pred / self.ro]
                ret['dets'].update({'preds': preds[0]})

        return output

