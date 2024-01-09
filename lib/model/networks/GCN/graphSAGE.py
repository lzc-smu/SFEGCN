"""graphSAGE to segment multi manual dataset"""
import torch
import torch.nn as nn
import networkx as nx
import dgl
from dgl.nn.pytorch import GraphConv


def ConstructConnectGraph(nx_g, col, sli):
    for k in range(sli):
        for j in range(col):
            if j == col - 1:
                nx_g.add_edge(k * col, k * col + j)
            else:
                nx_g.add_edge(k * col + j, k * col + j + 1)

            if j == col - 2:
                nx_g.add_edge(k * col, k * col + j)
            elif j == col - 1:
                nx_g.add_edge(k * col + 1, k * col + j)
            else:
                nx_g.add_edge(k * col + j, k * col + j + 2)

    for k in range(sli - 1):
        for j in range(col):
            nx_g.add_edge(k * col + j, (k + 1) * col + j)

    for k in range(sli - 1):
        for j in range(col):
            if j == col - 1:
                nx_g.add_edge(k * col + j, (k + 1) * col)
                nx_g.add_edge((k + 1) * col + j, k * col)
            else:
                nx_g.add_edge(k * col + j, (k + 1) * col + j + 1)
                nx_g.add_edge((k + 1) * col + j, k * col + j + 1)
    return nx_g


def construct_graph(infeat):
    nx_g = nx.DiGraph()
    sli, col = 1, infeat.size(2)        # 2D
    G = dgl.from_networkx(ConstructConnectGraph(nx_g, col, sli))
    g = dgl.to_bidirected(G)
    g = g.int().to(0)
    return g


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feat):
        super(GraphSAGE, self).__init__()
        self.gcn = nn.ModuleList()

        self.gcn = GraphConv(in_feats, out_feat, activation=nn.ReLU())
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs, graph):
        inputs = inputs.permute(0, 2, 1)
        sli, node, feat = inputs.size(0), inputs.size(1), inputs.size(2)

        # 2D
        output = torch.tensor([]).cuda()
        for i in range(sli):
            h = torch.reshape(inputs[i], [-1, feat])
            h = self.dropout(h)
            h = self.gcn(graph, h)
            output = torch.cat([output, h.unsqueeze(dim=0)], dim=0)
        output = output.permute(0, 2, 1)
        return output


class GCN(nn.Module):
    def __init__(self, state_dim, feature_dim):
        super(GCN, self).__init__()

        self.head = GraphSAGE(feature_dim, state_dim)

        self.res_layer_num = 4
        for i in range(self.res_layer_num):
            conv = GraphSAGE(state_dim, state_dim)
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x, graph):
        states = []

        x = self.head(x, graph)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x, graph) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x