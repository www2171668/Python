import torch
import torch.nn.functional as F
from layers import GraphConvolution

# model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        '''
           Build the gcn model: 1) first layer: nfeat->nhid; 2) second layer: nhid->nclass. 
           nfeat: the input feature number
           nhid: the hideen layer dimension
           nclass: the prediction results, i.e. c classes
           dropout: the dropout rate
        '''
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
