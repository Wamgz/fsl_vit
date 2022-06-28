import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from src.utils.logger_utils import logger
import torch.nn.functional as F

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
        x = rearrange(x, 'b n e -> b e n')
        return self.fn(rearrange(self.norm(x), 'b e n -> b n e'), **kwargs) # x: (600, 65, 1024)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Rearrange('b n e -> b e n'),
            nn.BatchNorm1d(dim),
            Rearrange('b n e -> b e n'),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_patch, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads # 1024  TODO，innerdim和dim有什么区别
        project_out = not (heads == 1 and dim_head == embed_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1) # tuple: ((600, 65, 1024), (600, 65, 1024), (600, 65, 1024))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv) # (batch, num_head, num_patch, head_dim) -> (600, 16, 65, inner_dim / head)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (batch, num_head, num_patch * num_patch, num_patch * num_patch)

        attn = self.attend(dots) # q和k的相似度矩阵, attn: (600, 16, 65, 65)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) # attn矩阵乘v不是点乘（对v加权），v的维度不变
        out = rearrange(out, 'b h n d -> b n (h d)') # (batch, num_patch, num_head * head_dim(inner_dim))

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads, dim_head, mlp_dim, num_patch, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ # 这里是先进行norm，再进行Attention和FFN
                PreNorm(embed_dim, Attention(embed_dim, num_patch=num_patch, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(embed_dim, FeedForward(embed_dim, mlp_dim, dropout=dropout))
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
    def __init__(self, *, image_size, patch_size, out_dim, embed_dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=12, tsfm_dropout=0., emb_dropout=0., feature_only=False, pretrained=False, patch_norm=True, conv_patch_embedding=False,
                 use_avg_pool_out=False, use_dual_feature=False):
        super().__init__()
        self.pretrained = pretrained

        image_height, image_width = pair(image_size) #
        patch_height, patch_width = pair(patch_size) # 32, 32

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


        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)
        # dim: 1024, depth: 6, heads: 16, dim_head: 64, mlp_dim: 2048, dropout: 0.1
        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, self.num_patch, tsfm_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim)
        )
        self.use_avg_pool_out = use_avg_pool_out
        self.norm = nn.LayerNorm(embed_dim) ## TODO 维度确定
        self.avg_pool = nn.AdaptiveAvgPool1d(1) ## TODO 维度确定

        self.out_head = nn.Sequential(
            nn.LayerNorm((self.num_patch + 1) * embed_dim),
            nn.Linear((self.num_patch + 1) * embed_dim, out_dim)
        )
        if pretrained:
            self.pretrained_model = timm.create_model('vit_base_patch16_224', num_classes=out_dim, pretrained=True)
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.head.parameters():
                param.requires_grad = True
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.apply(self._init_weights)
        self.bn = nn.BatchNorm2d(embed_dim)

        self.use_dual_feature = use_dual_feature
        self.avg_pool_64 = nn.AdaptiveAvgPool1d(64)
    def forward(self, img):
        if self.pretrained:
            return self.pretrained_model(img)

        # x: (batch, C, H, W) -> (600, 1, 256, 256)
        x = self.to_patch_embedding(img) # (batch, num_patch, patch_size * patch_size) -> (600, 64, 1024)
        if self.use_dual_feature:
            x_1 = self.to_patch_embedding(F.interpolate(img, [64, 64]))
            x_2 = self.to_patch_embedding(F.interpolate(img, [32, 32]))
            x = torch.cat((x, x_1, x_2), dim=1) # num_patch维度拼接
            x = self.avg_pool_64(x.transpose(1, 2)).transpose(1, 2)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b) # (batch, 1, embed_dim) -> (600, 1, 1024)
        x = torch.cat((cls_tokens, x), dim=1) # (batch, num_patch + 1, embed_dim) ->（600, 65, 1024）
        x += self.pos_embedding[:, :(n + 1)] # (batch, num_patch + 1, embedding_dim) ->（600, 65, 1024）
        x = self.dropout(x)

        x = self.transformer(x) # (batch, num_patch + 1, embedding_dim) -> (600, 65, 1024)
        if self.use_avg_pool_out:
            x = self.norm(x) # (batch, num_patch + 1, embedding_dim)
            x = self.avg_pool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
            return x
        else:
            x = x.view(b, -1)
            out = self.out_head(x)
            # out = self.layer_norm(out)
            return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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
            patch_size=16,
            out_dim=64,
            embed_dim=64,
            depth=4,
            heads=8,
            dim_head=8,
            mlp_dim=64,
            tsfm_dropout=0.1,
            emb_dropout=0.1,
            use_avg_pool_out=True,
            channels=3
        )

    x = torch.randn((400, 3, 96, 96))
    print(model(x).shape)
    # num_param = get_parameter_number(model)