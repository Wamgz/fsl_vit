# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from src.utils.parser_util import get_parser
from src.utils.logger_utils import logger
from models.gch import FeedForward
options = get_parser().parse_args()

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)

def dist_loss(x, y, type='euclidean'):
    '''
    Compute euclidean distance between two tensors
    x: query_samples: (300, 64)
    y: prototype: (60, 64)
    '''
    # x: N x D
    # y: M x D
    n = x.size(0) # (class_per_episode * num_query)
    m = y.size(0) # (class_per_episode)
    d = x.size(1) # (embed_dim)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d) # （class_per_episode * num_query, class_per_episode, embed_dim）(300, 60, 64)
    y = y.unsqueeze(0).expand(n, m, d) # (300, 60, 64)
    if type == 'euclidean':
        return torch.pow(x - y, 2).sum(2)
    elif type == 'cosine':
        scale = 100
        return scale * (1 - F.cosine_similarity(x, y, -1))


def prototypical_loss(model_outputs, labels, n_support, n_query, mlp=None, dist='euclidean', cuda=True, aux_loss=True, scale=1,
                      use_join_loss=True):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples (batch, H * W * z_dim)
    - target: ground truth for the above batch of samples (batch, )
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    def supp_idxs(c):
        # 从每个classes里取n_support（5）个input的索引出来，去input里取对应label的数据
        return labels.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(labels) # (classes_per_episode, )
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # 上面的dataset实际上模拟了一个episode(从整个train set取出的一个subset)，下面的support set和query set就是在这个episode中随机取一部分，loss计算也是计算query
    # 和support set的平均值作为prototype

    # support的每个类取前n_support个, 计算proto中心[tensor[0, 1, 2, 3, 4], tensor(5, 6, 10, 11, 12), ····]
    support_group_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([model_outputs[idx_list].mean(0) for idx_list in support_group_idxs]) # (class_per_episode, embedding_dim)

    # 取support和query的样本，用来计算距离中心的距离
    support_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[0:n_support], classes))).view(-1) # (class_per_episode * n_query)
    support_samples = model_outputs[support_idxs] # (class_per_episode * n_support, embedding_dim)
    query_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[n_support:], classes))).view(-1) # (n_classes * n_query)
    query_samples = model_outputs[query_idxs] # (class_per_episode * n_support, embedding_dim)
    labels = labels[torch.stack(list(map(lambda c: labels.eq(c).nonzero()[:], classes)))].view(-1)
    support_dists = dist_loss(support_samples, prototypes, dist) # (class_per_episode * n_support, class_per_episode)
    query_dists = dist_loss(query_samples, prototypes, dist) # (class_per_episode * n_query, class_per_episode)

    support_log_p_y = F.log_softmax(-support_dists, dim=-1).view(n_classes, n_support, -1) # (class_per_episode, n_support, class_per_episode)
    query_log_p_y = F.log_softmax(-query_dists, dim=-1).view(n_classes, n_query, -1) # (class_per_episode, n_support, class_per_episode)

    log_p_y = torch.cat((support_log_p_y, query_log_p_y), 1)
    target_inds = torch.arange(n_classes).view(n_classes, 1, 1).expand(n_classes, n_support + n_query, 1).long() # (class_per_episode, n_support + n_query, 1)
    if cuda:
        target_inds = target_inds.cuda()
    if not use_join_loss:
        log_p_y = query_log_p_y
        target_inds = torch.arange(n_classes).view(n_classes, 1, 1).expand(n_classes, n_query, 1).long()

    # 关键难以理解的地方：由于在sample的时候就是按照label取的，[0, 10)是第一个label，[10, 20)是第二个label，而计算loss的时候需要对应上，也就是第一个label的数据只需要保留和第一个prototype的距离
    # 同样第二个label的数据只需要保留和第二个prototype的数据
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)

    # logger.info('correct idx: {}'.format(y_hat))
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
    x_entropy = torch.tensor(0)
    if aux_loss:
        model_outputs = mlp(model_outputs)
        classification_hat = F.log_softmax(model_outputs, -1)
        x_entropy = F.cross_entropy(classification_hat, labels)
    return loss_val, scale * x_entropy,  acc_val


if __name__ == '__main__':
    model_outputs = torch.randn((100, 64))
    labels = torch.arange(5, 10).view(5, 1).expand(5, 20).reshape(-1)
    class_dict = {}
    for i in range(5):
        class_dict[i + 5] = i
    labels = labels[torch.randperm(100)]
    loss = prototypical_loss(model_outputs, labels, 5, 15, cuda=False)
    print(loss)