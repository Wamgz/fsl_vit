import torch
import torch.nn as nn
import numpy as np
class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        hdim = 640
        from networks.res12 import ResNet
        self.encoder = ResNet()

    def forward(self, x, support_idx, query_idx, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            if self.training:
                proto, logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
                return proto, logits, logits_reg
            else:
                logits = self._forward(instance_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')