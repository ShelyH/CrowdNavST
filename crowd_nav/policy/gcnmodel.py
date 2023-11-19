import torch.nn as nn
import torch.nn.functional as F

from crowd_nav.utils.layer import cheb_conv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, cheb_K, dropout, device):
        super(GCN, self).__init__()

        self.gc1 = cheb_conv(nfeat, nhid, cheb_K, device)
        self.gc2 = cheb_conv(nhid, nclass, cheb_K, device)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
