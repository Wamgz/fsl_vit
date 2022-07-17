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

    def forward(self, x):
        x = x.view(-1, 64, 5, 5)  # (100, 64, 5, 5)

        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs*1

        return out


# class Prototypical(nn.Module):
#     """Main Module for prototypical networlks"""
#     def __init__(self, args):
#         super(Prototypical, self).__init__()
#         self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
#         self.h_dim, self.z_dim = args['h_dim'], args['z_dim']
#
#         self.args = args
#         self.encoder = CNNEncoder()
#
#     def forward(self, inputs):
#         """
#             inputs are preprocessed
#             support:    (N_way*N_shot)x3x84x84
#             query:      (N_way*N_query)x3x84x84
#             s_labels:   (N_way*N_shot)xN_way, one-hot
#             q_labels:   (N_way*N_query)xN_way, one-hot
#         """
#         [support, s_labels, query, q_labels] = inputs
#         num_classes = s_labels.shape[1]
#         num_support = int(s_labels.shape[0] / num_classes)
#         num_queries = int(query.shape[0] / num_classes)
#
#         inp   = torch.cat((support,query), 0)
#         emb   = self.encoder(inp) # 80x64x5x5
#         emb_s, emb_q = torch.split(emb, [num_classes*num_support, num_classes*num_queries], 0)
#         emb_s = emb_s.view(num_classes, num_support, 1600).mean(1)
#         emb_q = emb_q.view(-1, 1600)
#         emb_s = torch.unsqueeze(emb_s,0)     # 1xNxD
#         emb_q = torch.unsqueeze(emb_q,1)     # Nx1xD
#         dist  = ((emb_q-emb_s)**2).mean(2)   # NxNxD -> NxN
#
#         ce = nn.CrossEntropyLoss().to(device)
#         loss = ce(-dist, torch.argmax(q_labels,1))
#         ## acc
#         pred = torch.argmax(-dist,1)
#         gt   = torch.argmax(q_labels,1)
#         correct = (pred==gt).sum()
#         total   = num_queries*num_classes
#         acc = 1.0 * correct.float() / float(total)
#
#         return loss, acc


class LabelPropagation(nn.Module):
    """Label Propagation"""

    def __init__(self):
        super(LabelPropagation, self).__init__()
        self.im_width, self.im_height, self.channels = (84, 84, 3)

        self.encoder = CNNEncoder()
        self.relation = RelationNetwork()
        self.rn = 30
        self.k = 20
        self.alpha = torch.tensor(0.99, requires_grad=True)

        self.cls_per_episode = 5
        self.num_support = 5
        self.num_query = 15
    def forward(self, imgs, labels):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        eps = np.finfo(float).eps
        labels = self._map2ZeroStart(labels)
        labels_unique, _ = torch.sort(torch.unique(labels))

        ## 拆分support和query，加上对应的class_embedding
        support_idxs, query_idxs = self._support_query_data(labels)

        [support, s_labels, query, q_labels] = imgs[support_idxs], labels[support_idxs], imgs[query_idxs], labels[query_idxs]
        s_labels, q_labels = torch.nn.functional.one_hot(s_labels, self.cls_per_episode), torch.nn.functional.one_hot(q_labels, self.cls_per_episode)
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)

        # Step1: Embedding
        inp = torch.cat((support, query), 0)  # (100, 3, 84, 84) 将suport和query set concat在一块
        emb_all = self.encoder(inp).view(-1, 1600)  # (100, 1600) 合并在一起提取特征
        N, d = emb_all.shape[0], emb_all.shape[1]

        # Step2: Graph Construction
        ## sigmma
        self.sigma = self.relation(emb_all)

        ## W
        emb_all = emb_all / (self.sigma + eps)  # N*d -> (100, 1600)
        emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
        emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
        W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N，实现wij = (fi - fj)**2
        W = torch.exp(-W / 2)

        ## keep top-k values
        topk, indices = torch.topk(W, self.k)  # topk: (100, 20), indices: (100, 20)
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
        yu = torch.zeros(num_classes * num_queries, num_classes)  # (75, 5)
        if torch.cuda.is_available():
            yu = yu.cuda()
        # yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).to(device)
        y = torch.cat((ys, yu), 0)  # (100, 5)用supoort set的label去预测query的label
        eye = torch.eye(N)
        if torch.cuda.is_available():
            eye = eye.cuda()
        s = torch.inverse(eye - self.alpha * S + eps)
        F = torch.matmul(s, y)  # (100, 5)
        Fq = F[num_classes * num_support:, :]  # query predictions，loss计算support和query set一起算，acc计算只计算query

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            ce = ce.cuda();
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
    support = torch.randn((25, 3, 84, 84))
    query = torch.randn((75, 3, 84, 84))
    imgs = torch.cat((support, query), 0)
    support_labels = torch.arange(5).view(1, -1).repeat(5, 1).view(-1) + 2
    query_labels = torch.arange(5).view(1, -1).repeat(15, 1).view(-1) + 2
    support_labels, query_labels = support_labels[torch.randperm(25)], query_labels[torch.randperm(75)]
    labels = torch.cat((support_labels, query_labels), 0)
    out = model(imgs, labels)

    print(out)
    # num_param = get_parameter_number(model)
