"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created foe EfficientNet, etc networks,however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion:https://github.com/tensorflow/tou.issues/494#issuecomment-532968956 ... I've opted for
    changing the ;ayer and argument names to 'drop path' rather tjam ,ox DropConnect as alayer name and use
    'survival rate' as the argument
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # nn.Identity() 输入是什么 输出就是什么 不作任何变化
        # 如果norm_layer是非None则会初始化一个norm_layer
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    # patch embedding
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model({self.img_size[0]}*{self.img_size[1]})."

        # transpose:[B, C, HW] -> [B, HW, C]
        # proj(x)出来为[B, 768, 14, 14]
        # flatten(0)从0维开始展开为[B*C*H*W] 从1维开始展开为[B, C*H*W]
        # flatten(2)展开后为[B, C, H*W] = [B, 768, 196]
        # transpose:[B, C, H*W] -> [B, HW, C] = [B, 196, 768]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# MultiHead Attention 模块
class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,  #
                 attn_drop_ration=0.,
                 proj_drop_ration=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 根号下dk分之一
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv是通过全连接层实现的
        self.attn_drop = nn.Dropout(attn_drop_ration)
        self.proj = nn.Linear(dim, dim)  # 将i个head进行concat后 通过矩阵Wo映射
        self.proj_drop = nn.Dropout(proj_drop_ration)

    def forward(self, x):
        # x的shape：[batch_size, num_patches+1, total_embed_dim]
        # num_patches+1 = 14 * 14 + 1(classToken) = 196 + 1 = 197
        B, N, C = x.shape
        # qkv() -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute调整数据顺序 ->[3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]

        qkv = self.qkv(x).reshape(B, N, 3, self.numm_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v, = qkv[0], qkv[1], qkv[2]
        # transpose -> [batch_size,  num_heads, embed_dim_per_head, num_patches + 1]
        # @ 矩阵乘法
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # softmax是对行进行处理
        '''
        transpose:只能选择tensor中两个维度进行转置
        k想象成barch_size个长方体 H和W是num_patches + 1和embed_dim_per_head 将这两个维度转置
        长方体切片成num_heads个 （num_patches + 1） * （embed_dim_per_head）的矩阵 
        切片后的矩阵进行转置 再与q做矩阵乘法
        permute：可以让tensor按照指定维度顺序（维度的个数就是该tensor的维度数）进行转置
        '''
        # 对每一个V进行加权求和
        # transpose -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape -> [batch_size, num_patches + 1, total_embed_dim] reshape将每个head concat到一切了
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 与Wo矩阵相乘
        x = self.proj(x)
        x = self.proj_drop(x)


# MLP Block
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
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


# Encoder Block
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,  # 第一个全连接层节点个数 是输入节点的4倍
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ration=0.,  # 多头注意力模块中最后的全连接层之后的dropout层对应的drop比率
                 attn_drop_ration=0.,  # 多头注意力模块中softmax[Q*k^t/根号dk]之后的dropout层的drop比率
                 drop_path_ration=0.,  # DropPath方法用到的比率
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        # 第一层LN
        self.norm1 = norm_layer(dim)
        # 第一个多头注意力
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ration=attn_drop_ration, proj_drop_ration=drop_ration)
        # 第二个LN层
        self.norm2 = norm_layer(dim)
        # mlp层的隐层个数是输入的4倍，实例化一个MLP模块的时候需要传入mlp_hidden_dim这个参数，所以在此提前计算
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ration)

    def forward(self, x):
        #  LN -> Multi-Head-Attention -> DropPath x从这三步输出后再加上第一个LN之前的输入x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # LN -> Mlp -> DropPath x从这三步后在家如第二个LN之前的输入x
        x = x + self.drop_path(self.drop_path(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_c=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,  # Encoder Block堆叠的次数
                 num_heads=12,
                 mlp_ration=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,  # 最后的MLP Head中pre-logins中全连接层的节点个数，
                 # 默认为None，此时就不会去构建这个pre-logins
                 distilled=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        # partial接收函数nn.LayerNorm作为参数 固定LayerNorm的参数eps=1e-6 并返回一个新的函数
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        '''
            nn.Parameter可以理解成一个类型转换函数，讲一个不可训练的类型Tensor转换成可以训练的类型Parameter，
        并将这个parameter绑定到module中，进过类型转换 self.x变成了模型的一部分，成为了模型中根据训练可以改动的参数了，
        使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
            在vit中 positional embedding和class token是两个随着网络训练学习的参数，但又不属于FC、MLP等运算的参数
        这时就可以使用nn.Parameter()来讲这个随机初始化的Tensor注册为可学习的参数Parameter。
        '''
        # torch.zeros 使用零矩阵初始化cls_token shape: 1 * 1 * embed_dim
        # 第一个1是batch_size维度为了后面的拼接设置成1 第二、三个维度是 1 * 768
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # 这一行可以直接忽略
        # 使用零矩阵初始化 position embedding
        self.pos_embed = nn.Parameter(torch.zeors(1, num_patches + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # 根据传入的drop_path_rate参数（默认为0），for i in的语句使得每一层的drop_path层的drop比率是递增的，但是默认为0，则不创建。
        # stochastic depth decay rule（随机深度衰减规则）
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # nn.Sequential的定义 输入要么是orderdict 要么是一系列的模型，对于列表list 必须用*进行转换
        self.block = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ration, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W]->[B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # torch.expand()将单个维度扩为更大尺寸
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # 在第二个维度上拼接
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])

    def forward(self, x):
        # 前向部分
        x = self.forward_features(x)  #
        if self.head_dist is not None:
            # 本模型head_dist=None（81行）所以不执行此分支 不用看
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)  # 直接来到这，head是255行定义的分类头
        return x



def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model