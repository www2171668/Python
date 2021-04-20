import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
from models import GCN

class TrainGCN():
    def __init__(self):
        #parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
        #parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
        myseed = 42
        self.epochs = 200
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden = 16
        self.dropout = 0.5
        np.random.seed(myseed)
        torch.manual_seed(myseed)

    def start_train(self):
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = utils.load_data()

        # Model and optimizer
        self.model = GCN(nfeat=features.shape[1], nhid=self.hidden, nclass=labels.max().item() + 1, dropout=self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            self.model.train()# 每次，开始时把model设置到train模式，从而来进行训练，这至少使得dropout起作用
            optimizer.zero_grad()
            # forward + backward + optimize
            output = self.model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = utils.accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            # 这里呢，把model设置到eval模式，从而来进行测试
            # Evaluate validation set performance separately, deactivates dropout during validation run.
            self.model.eval() 
            output = self.model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch+1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()), 'acc_val: {:.4f}'.format(acc_val.item()))


    def test(self):
        self.model.eval()
        adj, features, labels, idx_train, idx_val, idx_test = utils.load_data()
        output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

if __name__ == "__main__":
    myTrain = TrainGCN()
    t_total = time.time()
    myTrain.start_train()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    myTrain.test()

