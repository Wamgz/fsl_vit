import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from base import FewShotModel

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output
    
class FEAT(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        hdim = 640
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.args = args
    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
    
        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        proto = self.slf_attn(proto, proto, proto)        
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
        
        # for regularization
        if self.training:
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            if self.args.use_euclidean:
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
    
                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            
            
            return proto, logits, logits_reg
        else:
            return logits   

if __name__ == '__main__':
    import argparse
    import os
    def get_command_line_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_epoch', type=int, default=200)
        parser.add_argument('--episodes_per_epoch', type=int, default=100)
        parser.add_argument('--num_eval_episodes', type=int, default=600)
        parser.add_argument('--model_class', type=str, default='FEAT',
                            choices=['MatchNet', 'ProtoNet', 'BILSTM', 'DeepSet', 'GCN', 'FEAT', 'FEATSTAR', 'SemiFEAT',
                                     'SemiProtoFEAT'])  # None for MatchNet or ProtoNet
        parser.add_argument('--use_euclidean', action='store_true', default=False)
        parser.add_argument('--backbone_class', type=str, default='ConvNet',
                            choices=['ConvNet', 'Res12', 'Res18', 'WRN'])
        parser.add_argument('--dataset', type=str, default='MiniImageNet',
                            choices=['MiniImageNet', 'TieredImageNet', 'CUB'])

        parser.add_argument('--way', type=int, default=5)
        parser.add_argument('--eval_way', type=int, default=5)
        parser.add_argument('--shot', type=int, default=1)
        parser.add_argument('--eval_shot', type=int, default=1)
        parser.add_argument('--query', type=int, default=15)
        parser.add_argument('--eval_query', type=int, default=15)
        parser.add_argument('--balance', type=float, default=0)
        parser.add_argument('--temperature', type=float, default=1)
        parser.add_argument('--temperature2', type=float, default=1)  # the temperature in the

        # optimization parameters
        parser.add_argument('--orig_imsize', type=int,
                            default=-1)  # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--lr_mul', type=float, default=10)
        parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
        parser.add_argument('--step_size', type=str, default='20')
        parser.add_argument('--gamma', type=float, default=0.2)
        parser.add_argument('--fix_BN', action='store_true',
                            default=False)  # means we do not update the running mean/var in BN, not to freeze BN
        parser.add_argument('--augment', action='store_true', default=False)
        parser.add_argument('--multi_gpu', action='store_true', default=False)
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--init_weights', type=str, default=None)

        # usually untouched parameters
        parser.add_argument('--mom', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float,
                            default=0.0005)  # we find this weight decay value works the best
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--log_interval', type=int, default=50)
        parser.add_argument('--eval_interval', type=int, default=1)
        parser.add_argument('--save_dir', type=str, default='./checkpoints')

        return parser

    def postprocess_args(args):
        args.num_classes = args.way
        save_path1 = '-'.join([args.dataset, args.model_class, args.backbone_class,
                               '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query)])
        save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                               'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
                               str(args.lr_scheduler),
                               'T1{}T2{}'.format(args.temperature, args.temperature2),
                               'b{}'.format(args.balance),
                               'bsz{:03d}'.format(max(args.way, args.num_classes) * (args.shot + args.query)),
                               # str(time.strftime('%Y%m%d_%H%M%S'))
                               ])
        if args.init_weights is not None:
            save_path1 += '-Pre'
        if args.use_euclidean:
            save_path1 += '-DIS'
        else:
            save_path1 += '-SIM'

        if args.fix_BN:
            save_path2 += '-FBN'
        if not args.augment:
            save_path2 += '-NoAug'

        if not os.path.exists(os.path.join(args.save_dir, save_path1)):
            os.makedirs(os.path.join(args.save_dir, save_path1))
        args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
        return args


    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    model = FEAT(args)
    x = torch.randn((100, 3, 84, 84))
    print(model(x)[0].shape)
