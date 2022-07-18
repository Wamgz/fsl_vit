import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from src.data_loaders.prototypical_batch_sampler import PrototypicalBatchSampler
from src.utils.logger_utils import logger
import torch.nn.functional as F
import numpy as np
from src.utils.parser_util import get_parser
from src.datasets.miniimagenet import MiniImageNet
import tqdm
torch.set_printoptions(precision=None, threshold=999999, edgeitems=None, linewidth=None, profile=None)


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim) # 需要是float才行，long不行
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, class_dim=5, dropout=0.):
        super().__init__()
        self.dim = dim
        self.cls_embeddim = class_dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(self.dim)

    def forward(self, x):
        cls_token = x[:, :, -self.cls_embeddim:]

        x = self.fc1(x[:, :, :-self.cls_embeddim])
        x = rearrange(x, 'b n e -> b e n')
        x = self.bn(x)
        x = rearrange(x, 'b e n -> b n e')
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = torch.cat((x, cls_token), dim=-1)
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, class_embed_dim, num_patch, heads=8, dim_head=64, dropout=0., use_linear_v=False):
        super().__init__()
        inner_dim = dim_head * heads # 1024
        project_out = False
        self.embed_dim = embed_dim
        self.class_embed_dim = class_embed_dim
        self.heads = heads
        # self.scale = dim_head ** -0.5
        self.scale = 1

        self.num_patch = num_patch
        # self.attend = nn.Softmax(dim=-1)
        self.attend = nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, inner_dim, bias=False)

        self.apply(self._init_weights)
        if use_linear_v:
            self.to_v = nn.Linear(embed_dim + class_embed_dim, inner_dim + class_embed_dim, bias=False) # TODO 是否需要linear
        else:
            self.to_v = nn.Identity()
        self.alpha = torch.tensor(0.99, requires_grad=True)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim + class_embed_dim, embed_dim + class_embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        eps = torch.tensor(np.finfo(float).eps)
        if torch.cuda.is_available():
            eps = eps.cuda()


        # x: (batch, num_patch, embed_dim + class_embed_dim) -> (batch * num_patch, embed_dim + class_embed_dim)
        # q, k -> x[:, :, :embed_dim]
        batch, num_patch, dim = x.shape
        x = rearrange(x, 'b n d -> (b n) d') # (batch * num_patch, embed_dim + class_embed_dim)
        q, k = self.to_q(x[:, :-self.class_embed_dim]), self.to_k(x[:, :-self.class_embed_dim]) # tuple: ((batch * num_patch, inner_dim))
        v = self.to_v(x) # (batch * num_patch , inner_dim + class_embed_dim)
        # q, k = map(lambda t: rearrange(t, 'B (h d) -> h B d', h=self.heads), qk) # (num_head, batch * num_patch, head_dim)
        # v = rearrange(v, 'B (h d) -> h B d', h=self.heads) #  (num_head, batch * num_patch, head_dim)
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (num_head, batch * num_patch, batch * num_patch)
        q = torch.unsqueeze(q, 1)  # N*1*d
        k = torch.unsqueeze(k, 0)  # 1*N*d
        dots = ((q - k) ** 2).mean(2)  # N*N*d -> N*N，实现wij = (fi - fj)**2
        attn = self.attend(dots) # q和k的相似度矩阵, attn: (batch * num_patch, batch * num_patch)
        # val_max, _ = torch.max(dots, dim=-1)
        # val_min, _ = torch.min(dots, dim=-1)
        # attn = (dots - val_min) / (val_max - val_min)
        topk, indices = torch.topk(attn, 64)  # topk: (100, 20), indices: (100, 20)
        mask = torch.zeros_like(attn)
        if torch.cuda.is_available():
            mask = mask.cuda()
        mask = mask.scatter(1, indices, 1)  # (100, 100)
        mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # torch.t() 期望 input 为<= 2-D张量并转置尺寸0和1。   # union, kNN graph
        # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
        attn = attn * mask  # 构建无向图，上面的mask是为了保证把wij和wji都保留下来
        ## normalize
        N = attn.size(0)
        # D = attn.sum(0) # (100, )
        # D_sqrt_inv = torch.sqrt(1.0/(D+eps)) # (100, )
        # D1 = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N) # (100, 100)
        # D2 = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1) # (100, 100)
        # attn = D1*attn*D2
        eye = torch.eye(N)
        if torch.cuda.is_available():
            eye = eye.cuda()
        attn = torch.inverse(eye - self.alpha * attn + eps)

        # print('cls_token', rearrange(cls_token, '(b n) d -> b n d', b = batch, n = num_patch).mean(1))
        out = torch.matmul(attn, v)

        out = rearrange(out, '(b n) d -> b n d', b = batch, n = num_patch) # (batch, num_patch, inner_dim)

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, embed_dim, class_embed_dim, depth, heads, dim_head, mlp_dim, num_patch, dropout=0., use_linear_v=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ # 这里是先进行norm，再进行Attention和FFN
                PreNorm(embed_dim + class_embed_dim, Attention(embed_dim, class_embed_dim=class_embed_dim, num_patch=num_patch, heads=heads, dim_head=dim_head, dropout=dropout, use_linear_v=use_linear_v)),
                PreNorm(embed_dim + class_embed_dim, FeedForward(embed_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=84, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # （1, 3, 224, 224） -> (1, 96, 56 ,56) -> (1, 96, 56 * 56) -> (1, 3136, 96)B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, out_dim, embed_dim, depth, heads, mlp_dim, class_embed_dim=5, total_class=100, cls_per_episode=5,
                 support_num=5, query_num=15, pool='cls', channels=1, dim_head=12, tsfm_dropout=0., emb_dropout=0., feature_only=False, pretrained=False, patch_norm=True, conv_patch_embedding=False,
                 use_avg_pool_out=False, use_dual_feature=False, use_linear_v=False):
        super().__init__()
        self.pretrained = pretrained

        image_height, image_width = pair(image_size) #
        patch_height, patch_width = pair(patch_size) # 32, 32
        self.num_support, self.num_query = support_num, query_num
        self.cls_per_episode = cls_per_episode
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patch = (image_height // patch_height) * (image_width // patch_width) # 64
        patch_dim = channels * patch_height * patch_width # 3072
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.conv_patch_embedding = conv_patch_embedding
        if self.conv_patch_embedding:
            ## 卷积实现patch_embedding
            self.to_patch_embedding = PatchEmbed(
                img_size=image_size, patch_size=patch_size, in_chans=channels, embed_dim=embed_dim,
                norm_layer=nn.LayerNorm if patch_norm else None)
        else:
            ## MLP实现patch_embedding
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(patch_dim, embed_dim), # patch dim: 3072, dim: 1024
                nn.LayerNorm(embed_dim)
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch, embed_dim))
        nn.init.kaiming_normal_(self.pos_embedding)

        self.class_embed_dim = self.cls_per_episode

        # self.cls_token = nn.Parameter(torch.zeros(self.cls_per_episode + 1, self.class_embed_dim)) # patch维度的class_embed
        # torch.nn.init.orthogonal_(self.cls_token, gain=1)

        self.dropout = nn.Dropout(emb_dropout)
        # dim: 1024, depth: 6, heads: 16, dim_head: 64, mlp_dim: 2048, dropout: 0.1
        self.transformer = Transformer(embed_dim, class_embed_dim, depth, heads, dim_head, mlp_dim, self.num_patch, tsfm_dropout, use_linear_v=use_linear_v)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim)
        )
        self.use_avg_pool_out = use_avg_pool_out
        self.norm = nn.LayerNorm(embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.out_head = nn.Sequential(
            nn.LayerNorm((self.num_patch + 1) * embed_dim),
            nn.Linear((self.num_patch + 1) * embed_dim, out_dim)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.apply(self._init_weights)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.total_class = total_class
        self.use_dual_feature = use_dual_feature
        self.avg_pool_64 = nn.AdaptiveAvgPool1d(64)
    def forward(self, imgs, labels):
        '''
        :param imgs: (batch, C, H, W) -> (100, 3, 96, 96)
        :param labels: (batch, ) -> (100, )
        :return:
        '''
        ## patch embedding
        x = self.to_patch_embedding(imgs) # (batch, num_patch, patch_size * patch_size) -> (100, 12 * 12, 64)

        batch, num_patch, _ = x.shape
        x += self.pos_embedding[:, :num_patch] # (batch, num_patch, embed_dim)

        labels = self._map2ZeroStart(labels)
        labels_unique, _ = torch.sort(torch.unique(labels))

        ## 拆分support和query，加上对应的class_embedding
        support_idxs, query_idxs = self._support_query_data(labels)
        zeros = torch.zeros(query_idxs.size(0), num_patch, self.cls_per_episode)
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        support_cls_token, query_cls_token = torch.nn.functional.one_hot(labels[support_idxs], self.cls_per_episode), zeros # (num_support, class_per_episode)
        if torch.cuda.is_available():
            query_cls_token = query_cls_token.cuda()
        support_cls_tokens, query_cls_tokens = \
            support_cls_token.unsqueeze(1).repeat(1, self.num_patch, 1), query_cls_token # (num_support, num_patch, class_embed_dim)
        # support_x1, query_x1 = x[support_idxs], x[query_idxs]
        support_x, query_x = torch.cat((x[support_idxs], support_cls_tokens), dim=-1), torch.cat((x[query_idxs], query_cls_tokens), dim=-1) # patch维度拼接, (num_support, num_patch, embed_dim + embed_dim), (num_query, num_patch, embed_dim + embed_dim)
        x, labels = torch.cat((support_x, query_x), dim=0), torch.cat((labels[support_idxs], labels[query_idxs]))
        # print('init cls_token: ', x[:, :, -self.cls_per_episode:])
        ## transformer
        x = self.dropout(x)
        x = self.transformer(x) # (batch, num_patch, embedding_dim + class_embed_dim)

        ## 取出class_embed进行loss计算，(support和query）都计算
        logits = x[:, :, -self.class_embed_dim:] # (batch, num_patch, class_embed_dim)

        # x_entropy = nn.CrossEntropyLoss()
        # if torch.cuda.is_available():
        #     x_entropy = x_entropy.cuda()
        # labels_patch = labels.unsqueeze(1).repeat(1, self.num_patch).flatten(0)
        # loss = x_entropy(logits.view(-1, logits.size(-1)), labels_patch) # (batch, class_per_epi)
        #
        # _, max_indices = torch.max(logits, dim=-1)
        # mode,_ = torch.mode(max_indices)
        #
        # y_hat = mode[self.num_support * self.cls_per_episode:]
        # acc_val = y_hat.eq(labels[self.num_support * self.cls_per_episode:]).float().mean()
        return loss, acc_val


    def _support_query_data(self, labels):
        labels_unique, _ = torch.sort(torch.unique(labels))
        support_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[:self.num_support], labels_unique))).view(-1)  # (class_per_episode * num_support)
        query_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[self.num_support:], labels_unique))).view(-1)  # (class_per_episode * num_query)
        return support_idxs, query_idxs


    def _map2ZeroStart(self, labels):
        labels_unique, _ = torch.sort(torch.unique(labels))
        labels_index = torch.zeros(self.total_class)
        for idx, label in enumerate(labels_unique):
            labels_index[label] = idx
        for i in range(labels.size(0)):
            labels[i] = labels_index[labels[i]]
        return labels

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trainable_params(self):
        if self.pretrained:
            return self.pretrained_model.head.parameters()
        return self.parameters()

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    M = 1024 * 1024
    size = total_num / 4. / M
    print('参数量: %d\n模型大小: %.4fM' % (total_num, size))
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    model = ViT(
            image_size=96,
            patch_size=32,
            out_dim=64,
            embed_dim=64,
            depth=4,
            heads=8,
            dim_head=8,
            mlp_dim=64,
            tsfm_dropout=0.0,
            emb_dropout=0.0,
            use_avg_pool_out=True,
            channels=3
        )
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
    for batch in tr_iter:
        optim.zero_grad()
        x, y = batch  # x: (batch, C, H, W), y:(batch, )
        loss, acc = model(x, y)
        loss.backward()
        optim.step()
        print(loss, acc)