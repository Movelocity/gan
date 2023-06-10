import math
from inspect import isfunction
from functools import partial
from einops import rearrange
from torch import einsum

import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(x):
    return x is not None

# 有val时返回val，val为None时返回d
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 残差模块，将输入加到输出上
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# 上采样（反卷积）
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

# 下采样
def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, cond):
        device = cond.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = cond[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """Deep Residual Learning for Image Recognition"""
    
    def __init__(self, dim, dim_out, *, cond_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_emb_dim, dim_out))
            if exists(cond_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(cond_emb):
            cond_emb = self.mlp(cond_emb)
            h = rearrange(cond_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """A ConvNet for the 2020s"""

    def __init__(self, dim, dim_out, *, cond_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(cond_emb_dim, dim))
            if exists(cond_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(cond_emb):
            condition = self.mlp(cond_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Generator(nn.Module):
    def __init__(
        self,
        num_classes,
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_cond_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        if init_dim is None:
            init_dim = dim // 3 * 2

        dims = [*map(lambda m: int(dim * m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_class = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_class = partial(ResnetBlock, groups=resnet_block_groups)

        # cond embeddings
        if with_cond_emb:
            cond_dim = dim * 4

            # nn.Embedding(num_classes, cond_dim), 用 sine emb 取代 dict emb
            self.cond_mlp = nn.Sequential(
                nn.Embedding(num_classes, dim),
                nn.Linear(dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            cond_dim = None
            self.cond_mlp = None

        self.conv_pre = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.GELU(),
        )

        # layers
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_class(dim_out, dim_in, cond_emb_dim=cond_dim),
                        block_class(dim_in, dim_in, cond_emb_dim=cond_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        if out_dim is None:
            out_dim = channels
        self.final_conv = nn.Sequential(
            block_class(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, cond):
        # ( b, n, h, w )
        
        if exists(self.cond_mlp):
            c = self.cond_mlp(cond)

        x = self.conv_pre(x)
        # upsample. 通过一系列的上采样stage，每个stage都包含：
        #   2个 ResNet/ConvNeXT blocks + groupnorm + attention 
        #   + res + upsample
        for block1, block2, attn, upsample in self.ups:
            x = block1(x, c)
            x = block2(x, c)
            x = attn(x)
            x = upsample(x)

        # 最终，通过一个ResNet/ConvNeXT blocl和一个卷积层。
        # ( b, 3, h, w )
        return self.final_conv(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes,
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_cond_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.num_classes = num_classes

        dims = [*map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_class = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_class = partial(ResnetBlock, groups=resnet_block_groups)

        # cond embeddings
        self.cond_mlp = None
        
        self.conv_pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.GELU()
        )
        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_class(dim_in, dim_out),
                        block_class(dim_out, dim_out),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.post_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        # ( b, n, h, w )
        # 首先，输入通过一个卷积层，同时计算class t 所对应的embedding
        # downsample. 通过一系列的下采样stage，每个stage都包含：
        #   2个 ResNet/ConvNeXT blocks + groupnorm + attention 
        #   + res + downsample
        x = self.conv_pre(x)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            x = downsample(x)
        x = self.post_layers(x)
        return x

if __name__ == "__main__":
    gen = Generator(dim=64)

    t1 = torch.zeros(1, 512, 32, 32)
    c = torch.Tensor([3])
    out = gen(t1, c)
    out.shape

    disc = Discriminator(dim=64)
