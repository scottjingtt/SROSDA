import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class GenZ(nn.Module):
    def __init__(self, input_size=2048, h1=1024, h2=512, normalize=True):
        super(GenZ, self).__init__()
        self.normalize = normalize
        self.layer1 = nn.Sequential(nn.Linear(input_size, h1), nn.LeakyReLU(0.2, inplace=True))#, nn.Dropout(0.4)), nn.BatchNorm1d(h1)
        self.layer2 = nn.Sequential(nn.Linear(h1, h2), nn.ReLU()) #, nn.Tanh())

    def forward(self, x):
        x = self.layer1(x)
        out = self.layer2(x)
        return out

class GenA(nn.Module):
    def __init__(self, input_size=512, h1=256, h2=85):
        super(GenA, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, h1), nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(h1, h2), nn.Sigmoid())

    def forward(self, x, adj=None):
        x = self.layer1(x)
        out = self.layer2(x)
        return out

# class DmDis(nn.Module):
#     def __init__(self, input_size=512+85, h1=256, h2=1):
#         super(DmDis, self).__init__()
#         self.layer1 = nn.Sequential(nn.Linear(input_size, h1), nn.LeakyReLU(0.2, inplace=True))#nn.BatchNorm1d(h1), , nn.Dropout(0.4))
#         self.layer2 = nn.Sequential(nn.Linear(h1, h2), nn.Sigmoid())
#     def forward(self, x):
#         x = self.layer1(x)
#         out = self.layer2(x)
#         return out


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features=512, out_features=85, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat=512, nhid=256, nclass=85, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return self.sigmoid(x)


class Clf(nn.Module):
    def __init__(self, input_size=512, h1=256, h2=40+1):
        super(Clf, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, h1), nn.LeakyReLU(0.2, inplace=True))#nn.BatchNorm1d(h1), , nn.Dropout(0.4))
        self.layer2 = nn.Sequential(nn.Linear(h1, h2))
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        logit = self.layer2(x)
        prob = self.softmax(logit)
        pred = torch.argmax(prob, dim=1)
        mul_dis = self.sigmoid(logit)
        return prob, pred, mul_dis

class ClfSU(nn.Module):
    def __init__(self, input_size=512, h1=256, h2=2):
        super(ClfSU, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.input_size = input_size
        self.layer1 = nn.Sequential(nn.Linear(input_size,h1), nn.LeakyReLU(0.2, inplace=True))#nn.BatchNorm1d(h1), , nn.Dropout(0.4))
        self.layer2 = nn.Sequential(nn.Linear(h1, h2), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        prob = self.layer2(x)
        pred = torch.argmax(prob, dim=1)
        return prob, pred


class ProtClf(nn.Module):
    def __init__(self):
        super(ProtClf, self).__init__()
        self.eps = 1e-12

    def forward(self, z, c, dist='cosine', T=0.1):
        if dist == 'cosine':
            norm = torch.mm(torch.norm(z, p=2, dim=1).unsqueeze(1), torch.norm(c, p=2, dim=1).unsqueeze(0))
            sim = torch.mm(z, c.t()) / torch.max(norm, self.eps * torch.ones_like(norm))
        elif dist == 'euclidean':
            dist_map = torch.cdist(z, c, p=2)
            sim = 1 - dist_map
        else:
            raise ValueError("dist does not exist!")
        y_prob = F.softmax(sim / T, dim=1)
        y_pred = torch.argmax(y_prob, dim=1)
        return y_prob, y_pred


def build_model(args, networks='fc', prot=None, dist=None, input_size=512, h1=256, h2=40):
    if networks == 'GenZ':
        return GenZ(input_size=input_size, h1=h1, h2=h2)
    elif networks == 'GenA':
        return GenA(input_size=input_size, h1=h1, h2=h2)
    elif networks == 'Clf':
        return Clf(input_size=input_size, h1=h1, h2=h2)
    elif networks == 'ClfSU':
        return ClfSU(input_size=input_size, h1=h1, h2=h2)
    elif networks == 'Prot':
        return ProtClf()
    elif networks == 'GCN':
        return GCN(nfeat=input_size, nhid=h1, nclass=h2, dropout=0.5)
    else:
        raise ValueError('Unrecognized networks type ', networks)
