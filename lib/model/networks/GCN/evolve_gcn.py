import torch.nn as nn
import torch
from .graphSAGE import GCN, construct_graph
from . import gcn_utils


class ResGCN(nn.Module):
    def __init__(self, opt):
        super(ResGCN, self).__init__()

        self.opt = opt
        self.points = self.opt.gcn_points
        self.fuse = nn.Conv1d(self.points, 64, 1)
        self.init_gcn = GCN(state_dim=self.points, feature_dim=64+2)
        self.evolve_gcn = GCN(state_dim=self.points, feature_dim=64+2)
        self.iter = 2
        self.ro = 4
        for i in range(self.iter):
            evolve_gcn = GCN(state_dim=self.points, feature_dim=64+2)
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_cirpoint(self, output, batch):
        init = gcn_utils.init_cirpoint(output, batch)
        output.update({'i_it_py': init['i_it_py'], 'i_gt_py': init['i_gt_py'], 'pre_gt_py': init['pre_gt_py']})
        return init

    def test_cirpoint(self, output):
        evolve = gcn_utils.test_cirpoint(output['dets'], self.points, self.opt.out_thresh)
        output.update({'it_py': evolve['i_it_py']})
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

            pred  = self.evolve_contour(self.evolve_gcn, cnn_feature, init['i_it_py'], init['c_it_py'], init['py_ind'])
            preds = [pred]
            for i in range(self.iter):
                pred = pred / self.ro
                c_pred = gcn_utils.img_poly_to_can_poly(pred)
                resgcn = self.__getattr__('resgcn'+str(i))
                pred = self.evolve_contour(resgcn, cnn_feature, pred, c_pred, init['py_ind'])
                preds.append(pred)
            ret.update({'py_pred': preds, 'i_gt_ctrs': output[0]['i_gt_ctrs'] * self.ro})

        if batch is None:
            with torch.no_grad():
                init_contour = self.test_cirpoint(output[0])
                ind = init_contour['ind']
                pred = self.evolve_contour(self.resgcn, cnn_feature, init_contour['i_it_py'], init_contour['c_it_py'], ind)
                for i in range(self.iter):
                    pred = pred / self.ro
                    c_pred = gcn_utils.img_poly_to_can_poly(pred)
                    resgcn = self.__getattr__('resgcn'+str(i))
                    py = self.evolve_contour(resgcn, cnn_feature, pred, c_pred, ind)
                    preds = [pred / self.ro]
                ret['dets'].update({'preds': preds[0]})

        return output

