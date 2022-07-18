# -------------------------------------
# Project: Transductive Propagation Network for Few-shot Learning
# Date: 2019.1.11
# Author: Yanbin Liu
# All Rights Reserved
# -------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from src.data_loaders.prototypical_batch_sampler import PrototypicalBatchSampler
from src.utils.logger_utils import logger
import torch.nn.functional as F
import numpy as np
from src.utils.parser_util import get_parser
from src.datasets.miniimagenet import MiniImageNet


class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))

    def forward(self, x):
        """x: bs*3*84*84 """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class RelationNetwork(nn.Module):
    """Graph Construction Module"""

    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2 * 2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding

    def forward(self, x, rn):
        x = x.view(-1, 64, 5, 5)  # (100, 64, 5, 5)

        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs*1

        return out


class LabelPropagation(nn.Module):
    """Label Propagation"""

    def __init__(self):
        super(LabelPropagation, self).__init__()
        self.im_width, self.im_height, self.channels = 84, 84, 3

        self.encoder = CNNEncoder()
        self.relation = RelationNetwork()

        self.alpha = torch.tensor(0.99)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        eps = np.finfo(float).eps

        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)

        # Step1: Embedding
        inp = torch.cat((support, query), 0)  # (100, 3, 84, 84) 将suport和query set concat在一块
        emb_all = self.encoder(inp).view(-1, 1600)  # (100, 1600) 合并在一起提取特征
        N, d = emb_all.shape[0], emb_all.shape[1]

        # Step2: Graph Construction
        ## sigmma

        self.sigma = self.relation(emb_all, 30)

        ## W
        emb_all = emb_all / (self.sigma + eps)  # N*d -> (100, 1600)
        emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
        emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
        W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N，实现wij = (fi - fj)**2
        W = torch.exp(-W / 2)

        ## keep top-k values

        topk, indices = torch.topk(W, 20)  # topk: (100, 20), indices: (100, 20)
        mask = torch.zeros_like(W)
        mask = mask.scatter(1, indices, 1)  # (100, 100)
        mask = ((mask + torch.t(mask)) > 0).type(
            torch.float32)  # torch.t() 期望 input 为<= 2-D张量并转置尺寸0和1。   # union, kNN graph
        # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
        W = W * mask  # 构建无向图，上面的mask是为了保证把wij和wji都保留下来

        ## normalize
        D = W.sum(0)  # (100, )
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))  # (100, )
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)  # (100, 100)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)  # (100, 100)
        S = D1 * W * D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels  # (25, 5)
        yu = torch.zeros(num_classes * num_queries, num_classes) # (75, 5)
        if torch.cuda.is_available():
            yu = yu.cuda()
        # yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).to(device)
        y = torch.cat((ys, yu), 0)  # (100, 5)用supoort set的label去预测query的label
        eye = torch.eye(N)
        if torch.cuda.is_available():
            eye = eye.cuda()
        s = torch.inverse(torch.eye(N) - self.alpha * S + eps)
        F = torch.matmul(s, y)  # (100, 5)
        Fq = F[num_classes * num_support:, :]  # query predictions，loss计算support和query set一起算，acc计算只计算query

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            ce = ce.cuda()
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()
        total = num_queries * num_classes
        acc = 1.0 * correct.float() / float(total)

        return loss, acc

    def trainable_params(self):
        return self.parameters()

    def _map2ZeroStart(self, labels):
        labels_unique, _ = torch.sort(torch.unique(labels))
        labels_index = torch.zeros(100)
        for idx, label in enumerate(labels_unique):
            labels_index[label] = idx
        for i in range(labels.size(0)):
            labels[i] = labels_index[labels[i]]
        return labels

    def _support_query_data(self, labels):
        labels_unique, _ = torch.sort(torch.unique(labels))
        support_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[:self.num_support], labels_unique))).view(
            -1)  # (class_per_episode * num_support)
        query_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[self.num_support:], labels_unique))).view(
            -1)  # (class_per_episode * num_query)
        return support_idxs, query_idxs
if __name__ == '__main__':
    model = LabelPropagation()
    # region
    def init_dataset(opt, mode):
        dataset = MiniImageNet(mode=mode, opt=options)
        _dataset_exception_handle(dataset=dataset, n_classes=len(np.unique(dataset.y)), mode=mode, opt=opt)
        return dataset
    def _dataset_exception_handle(dataset, n_classes, mode, opt):
        n_classes = len(np.unique(dataset.y))
        if mode == 'train' and n_classes < opt.classes_per_it_tr or mode == 'val' and n_classes < opt.classes_per_it_val:
            raise (Exception('There are not enough classes in the data in order ' +
                             'to satisfy the chosen classes_per_it. Decrease the ' +
                             'classes_per_it_{tr/val} option and try again.'))
    def init_sampler(opt, labels, mode, dataset_name='miniImagenet'):
        num_support, num_query = 0, 0
        if 'train' in mode:
            classes_per_it = opt.classes_per_it_tr
            num_support, num_query = opt.num_support_tr, opt.num_query_tr
        else:
            classes_per_it = opt.classes_per_it_val
            num_support, num_query = opt.num_support_val, opt.num_query_val

        return PrototypicalBatchSampler(labels=labels,
                                        classes_per_it=classes_per_it,
                                        num_support=num_support,
                                        num_query=num_query,
                                        iterations=opt.iterations)
    def init_dataloader(opt, mode):
        dataset = init_dataset(opt, mode)
        sampler = init_sampler(opt, dataset.y, mode)

        dataloader_params = {
            # 'pin_memory': True,
            # 'num_workers': 8
        }
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, **dataloader_params)
        return dataset, dataloader
    # endregion

    options = get_parser().parse_args()
    print('option', options)
    tr_dataset, tr_dataloader = init_dataloader(options, 'train')
    tr_iter = iter(tr_dataloader)
    tr_iter.__next__()
    optim = torch.optim.Adam(model.trainable_params(), lr=0.001)
    model.train()
    train_loss = []
    train_acc = []
    for batch in tr_iter:
        optim.zero_grad()
        x, y = batch  # x: (batch, C, H, W), y:(batch, )
        loss, acc = model(x, y)
        loss.backward()
        optim.step()
        train_loss.append(loss.detach())
        train_acc.append(acc.detach())
        print(loss, acc)
    train_avg_loss = torch.tensor(train_loss[:]).mean()
    train_avg_acc = torch.tensor(train_acc[:]).mean()