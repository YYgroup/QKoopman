from torch.utils.checkpoint import checkpoint
import numpy as np
import h5py
import math
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

import json
import re
import time # 计时



# ==================== 模型相关模块 ====================
class ResidualBlock(nn.Module):
    '''残差块
    '''
    def __init__(self, in_chans, out_chans, stride=1, groups=1, act_layer=nn.SiLU, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=True) # 3x3 卷积
        self.bn1 = nn.BatchNorm2d(out_chans)

        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True)  # 3x3 分组卷积
        self.bn2 = nn.BatchNorm2d(out_chans)

        self.act = act_layer()  # 激活函数
        self.dropout = nn.Dropout(drop)  # Dropout 正则化

        # 如果输入和输出的维度不一致，使用 1x1 卷积调整维度
        self.downsample = None
        if stride != 1 or in_chans != out_chans:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans)
            )

    def forward(self, x):
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        # 如果维度不一致，调整输入的维度
        if self.downsample is not None:
            x = self.downsample(x)
        # 残差连接 + Dropout
        out = out + x
        out = self.dropout(out)
        out = self.act(out)  # 最终激活
        return out


class DWConv(nn.Module):
    '''MixFFN的子模块,输入输出形状相同->(B,H*W,dim)
    '''
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim) # 深度分组卷积

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class MixFFN(nn.Module):
    '''AttentionBlock的子模块,输入(B,H*W,in_features),输出(B,H*W,out_features)
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    '''AttentionBlock的子模块,输入输出形状相同->(B,H*W,dim)
    论文 SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    降低注意力模块计算复杂度,Efﬁcient Self-Attention,论文中的R等于代码的sr_ratio^2
    K&V.shape : (B,num_heads,H*W/sr_ratio^2,d_head)
    Q.shape : (B,num_heads,H*W,d_head)
    zby修改: 1. 使用深度可分离卷积执行降低分辨率以避免操作本身所需要的参数量占主导
    zby修改: 2. 支持调整d_model参数
    '''
    def __init__(self, dim, d_model=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, sr_groups=None):
        super().__init__()
        self.d_model = d_model or dim
        assert self.d_model % num_heads == 0, f"d_model {dim} should be divided by num_heads {num_heads}."
        self.d_head = self.d_model // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.scale = qk_scale or (self.d_model // num_heads) ** -0.5

        self.q = nn.Linear(dim, self.d_model, bias=qkv_bias)
        self.kv = nn.Linear(dim, self.d_model * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.d_model, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_groups = sr_groups or dim
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio,groups=self.sr_groups)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape # N=H*W
        q = self.q(x).view(B, N, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).view(B, C, H, W)
            x_ = self.sr(x_).view(B, C, -1).permute(0, 2, 1) # (B,C,H/sr_ratio,W/sr_ratio)->(B,H*W/sr_ratio^2,C)
            x_ = self.norm(x_)
            kv = self.kv(x_).view(B, -1, 2, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).view(B, -1, 2, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().view(B, N, self.d_model)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PreNorm(nn.Module):
    '''AttentionBlock的子模块
    返回的只是fn(norm(x)),而prenorm为x+fn(norm(x))
    此子块没有进行残差连接
    '''
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class AttentionBlock(nn.Module):
    '''DCTBlock的子模块,输入(B,H*W,in_dim),输出(B,H*W,dim)
    '''
    def __init__(self,in_dim, dim, num_heads=8, d_model=None, ffn_ratio=2, attn_drop=0., proj_drop=0., ffn_drop=0.):
        super().__init__()
        self.mlp = nn.Linear(in_features=in_dim,out_features=dim)
        self.attn = PreNorm(dim,Attention(dim=dim,d_model=d_model,num_heads=num_heads,attn_drop=attn_drop,proj_drop=proj_drop))
        self.mixffn = PreNorm(dim,MixFFN(in_features=dim,hidden_features=ffn_ratio*dim,out_features=dim,drop=ffn_drop))
    def forward(self, x, H, W):
        x = self.mlp(x)
        x = x + self.attn(x,H=H,W=W)
        x = x + self.mixffn(x,H=H,W=W)
        return x


class DCTBlock(nn.Module):
    '''FEBlock的子模块,输入输出形状相同->(B,H*W,features)
    '''
    def __init__(self,features,growth_rate=32,attn_depth=4,ffn_ratio=2,num_heads=8,d_model=None,
                 attn_drop=0.5, proj_drop=0.5, ffn_drop=0.5):
        super().__init__()
        self.attnblocks = nn.ModuleList([])
        assert (d_model or growth_rate) % num_heads == 0, f"DCTBlock: (d_model or growth_rate) {d_model or growth_rate} should be divided by num_heads {num_heads}."
        for i in range(attn_depth):
            self.attnblocks.append(
                AttentionBlock(in_dim=features+i*growth_rate,
                                dim=growth_rate,d_model=d_model,num_heads=num_heads,
                                attn_drop=attn_drop,proj_drop=proj_drop,
                                ffn_ratio=ffn_ratio,ffn_drop=ffn_drop
                            ))
        self.out_layer = MixFFN(in_features=features + attn_depth * growth_rate,
                                hidden_features=None, # None -> hidden_features = in_features
                                out_features=features,
                                drop=ffn_drop)

    def forward(self,x,H,W):
        '''使用 `torch.utils.checkpoint` 包装前向传播，以减少显存占用。
        将原始的前向传播逻辑移至 `_forward_src` 方法，并通过 `checkpoint` 调用。
        '''      
        return checkpoint(self._forward_src,x,H,W,use_reentrant=False)

    def _forward_src(self,x,H,W):
        features = [x]
        for layer in self.attnblocks:
            x = torch.cat(features, 2)
            x = layer(x,H=H,W=W)
            features.append(x)
        x = torch.cat(features, 2)
        x = self.out_layer(x,H=H,W=W)
        return x


class OverlapEmbedding(nn.Module):
    '''FEBlock的子模块,输入(B,in_chans,img_size,img_size),输出(B,H*W,embed_dim)
    Image to Patch Embedding
    Overlap : stride < patch_size
    '''
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=(patch_size//2) )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x) # x.shape (batch_size,H*W,embed_dim)
        return x, H, W

class FEBlock(nn.Module):
    '''特征嵌入块,输入(B,in_chans,img_size,img_size),
    经过OverlapEmbedding后得到(B,H*W,features),
    经过DCTBlock后得到(B,H*W,features),
    经过view后得到(B,features,H,W)
    patch_size为奇数 -> H = W = (img_size-1)//stride + 1
    '''
    def __init__(self,in_chans,features,patch_size,stride,
                 growth_rate=32,attn_depth=4,ffn_ratio=2,
                 dct_depth=2,num_heads=8,d_model=None,
                 attn_drop=0.5, proj_drop=0.5, ffn_drop=0.5):
        super().__init__()
        self.embed=OverlapEmbedding(in_chans=in_chans,embed_dim=features,patch_size=patch_size,stride=stride)
        self.dctblocks = nn.ModuleList([])
        for _ in range(dct_depth):
            self.dctblocks.append(
                DCTBlock(features=features,growth_rate=growth_rate,attn_depth=attn_depth,
                         num_heads=num_heads,d_model=d_model,ffn_ratio=ffn_ratio,
                         attn_drop=attn_drop, proj_drop=proj_drop, ffn_drop=ffn_drop)
                        )

    def forward(self,x):
        '''使用 `torch.utils.checkpoint` 包装前向传播，以减少显存占用。
        将原始的前向传播逻辑移至 `_forward_src` 方法，并通过 `checkpoint` 调用。
        '''      
        return checkpoint(self._forward_src,x,use_reentrant=False)

    def _forward_src(self,x):
        B = x.size(0) # batch_size
        x,H,W = self.embed(x)
        for layer in self.dctblocks:
            x = layer(x,H=H,W=W)
        return x.view(B,-1,H,W)


class BasicConv2d(nn.Module):
    '''卷积块:Conv-Norm-act
    '''
    def __init__(self, in_chans, out_chans, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, **kwargs),
            nn.BatchNorm2d(out_chans),
            nn.SiLU(inplace=True) # inplace=True 可以节约显存
        )

    def forward(self, x):
        return self.conv(x)


class UnitBlock(nn.Module):
    '''KNO块中的酉算符 (优化版)
    优化：避免生成大矩阵、利用广播机制、减少中间变量
    '''
    def __init__(self, features):
        super().__init__()
        self.features = int(features)
        self.weight_len = int(torch.log2(torch.tensor(features)))
        self.weight = nn.Parameter(torch.zeros(self.weight_len))  # 0初始化
        # 预计算二进制位权重基向量 (优化)
        self.register_buffer(
            "binary_bases",
            2 ** (self.weight_len - 1 - torch.arange(self.weight_len)),
            persistent=False
        )
    def forward(self, input_phi, p=torch.tensor(1.)):
        # 优化：直接生成最终结果，避免中间矩阵
        z = self._generate_value_optimized()
        return input_phi + p.view(-1,1) * z.view(1,-1)  # 保持维度兼容
    def _generate_value_optimized(self):
        """优化后的值生成方法 (显存降低90%+)"""
        # 优化：逐位计算替代矩阵乘法
        binary_mask = (torch.arange(self.features, device=self.weight.device).unsqueeze(1) // self.binary_bases) % 2
        z = torch.sum(binary_mask * self.weight, dim=1) - torch.sum(self.weight) / 2
        return z
    def _init_weights(self):
        nn.init.constant_(self.weight, 0.0)  # 0初始化    


class KNOBlock(nn.Module):
    '''KNO块
    '''
    def __init__(self,c,h,w):
        super().__init__()
        self.c = int(c)
        self.h = int(h)
        self.w = int(w)
        self.features = int(c * h * w)
        self.kno=UnitBlock(self.features)

    def forward(self,x,p=torch.tensor(1.)):
        b = x.shape[0]
        x = x.view(b,self.features)
        x = self.kno(x,p=p)
        x = x.view(b,self.c,self.h,self.w)
        return x

class Encoder5(nn.Module):
    def __init__(self,in_channels=4,n_filters=4,
                 res_drop=0.5,attn_drop=0.5,proj_drop=0.5,ffn_drop=0.5,):
        super().__init__()
        self.block11=ResidualBlock(in_chans=in_channels, out_chans=n_filters,
                            stride=1, groups=1, drop=res_drop)
        self.block21=BasicConv2d(in_chans=in_channels,out_chans=in_channels,
                            kernel_size=3, stride=1, padding=1)
        self.block22=ResidualBlock(in_chans=in_channels, out_chans=2*n_filters,
                            stride=2, groups=1, drop=res_drop)
        self.block31=FEBlock(in_chans=2*n_filters,features=4*n_filters,
                            patch_size=3,stride=2,growth_rate=4*n_filters,
                            attn_depth=4,ffn_ratio=2,dct_depth=2,
                            num_heads=8,d_model=8*n_filters,
                            attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop
                            )
        self.block41=FEBlock(in_chans=4*n_filters,features=8*n_filters,
                            patch_size=3,stride=2,growth_rate=4*n_filters,
                            attn_depth=4,ffn_ratio=2,dct_depth=2,
                            num_heads=8,d_model=8*n_filters,
                            attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop
                            )
        self.block51=FEBlock(in_chans=8*n_filters,features=16*n_filters,
                            patch_size=3,stride=2,growth_rate=4*n_filters,
                            attn_depth=4,ffn_ratio=2,dct_depth=2,
                            num_heads=8,d_model=8*n_filters,
                            attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop
                            )
        self.down12=nn.Conv2d(n_filters,2*n_filters,kernel_size=3,stride=2,padding=1)
        self.down23=nn.Conv2d(2*n_filters,4*n_filters,kernel_size=3,stride=2,padding=1)
        self.down34=nn.Conv2d(4*n_filters,8*n_filters,kernel_size=3,stride=2,padding=1)
        self.down45=nn.Conv2d(8*n_filters,16*n_filters,kernel_size=3,stride=2,padding=1)

        self.block23=BasicConv2d(in_chans=2*n_filters,out_chans=2*n_filters,kernel_size=3,stride=1,padding=1)
        self.block32=BasicConv2d(in_chans=4*n_filters,out_chans=4*n_filters,kernel_size=3,stride=1,padding=1)
        self.block42=BasicConv2d(in_chans=8*n_filters,out_chans=8*n_filters,kernel_size=3,stride=1,padding=1)
        self.block52=BasicConv2d(in_chans=16*n_filters,out_chans=16*n_filters,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        # x11
        x11 = self.block11(x)

        # x23
        x21 = self.block21(x)
        x22 = self.block22(x21)
        x23 = self.block23(self.down12(x11)+x22)

        # x32
        x31 = self.block31(x22)
        x32 = self.block32(x31+self.down23(x23))

        # x42
        x41 = self.block41(x31)
        x42 = self.block42(x41+self.down34(x32))

        # x52
        x51 = self.block51(x41)
        x52 = self.block52(x51+self.down45(x42))
        return [x11,x23,x32,x42,x52]


class Decoder5(nn.Module):
    def __init__(self,out_channels=1,n_filters=4,res_drop=0.5):
        super().__init__()
        self.up54=nn.ConvTranspose2d(16*n_filters,8*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.up43=nn.ConvTranspose2d(8*n_filters,4*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.up32=nn.ConvTranspose2d(4*n_filters,2*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.up21=nn.ConvTranspose2d(2*n_filters,1*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)

        self.block43=BasicConv2d(in_chans=16*n_filters,out_chans=16*n_filters,kernel_size=3,stride=1,padding=1)
        self.block44=ResidualBlock(in_chans=16*n_filters, out_chans=8*n_filters,
                            stride=1, groups=1, drop=res_drop)
        # self.block45=nn.Conv2d(in_channels=8*n_filters,out_channels=8*n_filters,kernel_size=1,stride=1,padding=0) # 1*1的卷积核

        self.block33=BasicConv2d(in_chans=8*n_filters,out_chans=8*n_filters,kernel_size=3,stride=1,padding=1)
        self.block34=ResidualBlock(in_chans=8*n_filters, out_chans=4*n_filters,
                            stride=1, groups=1, drop=res_drop)
        # self.block35=nn.Conv2d(in_channels=4*n_filters,out_channels=4*n_filters,kernel_size=1,stride=1,padding=0) # 1*1的卷积核

        self.block24=BasicConv2d(in_chans=4*n_filters,out_chans=4*n_filters,kernel_size=3,stride=1,padding=1)
        self.block25=ResidualBlock(in_chans=4*n_filters, out_chans=2*n_filters,
                            stride=1, groups=1, drop=res_drop)
        # self.block26=nn.Conv2d(in_channels=2*n_filters,out_channels=2*n_filters,kernel_size=1,stride=1,padding=0) # 1*1的卷积核

        self.block12=BasicConv2d(in_chans=2*n_filters,out_chans=2*n_filters,kernel_size=3,stride=1,padding=1)
        self.block13=nn.Conv2d(in_channels=2*n_filters,out_channels=out_channels,kernel_size=1,stride=1,padding=0) # 1*1的卷积核


    def forward(self,x11,x23,x32,x42,x52):
        x43 = self.block43(torch.cat([x42,self.up54(x52)],dim=1))
        x44 = self.block44(x43)
        x33 = self.block33(torch.cat([x32,self.up43(x44)],dim=1))
        x34 = self.block34(x33)
        x24 = self.block24(torch.cat([x23,self.up32(x34)],dim=1))
        x25 = self.block25(x24)
        x12 = self.block12(torch.cat([x11,self.up21(x25)],dim=1))
        x13 = self.block13(x12)
        return x13

class MyModel5(nn.Module):
    def __init__(self,img_size=128,in_channels=3,out_channels=1,n_filters=16,
                 res_drop=0.5,attn_drop=0.5,proj_drop=0.5,ffn_drop=0.5,):
        super().__init__()
        self.encr = Encoder5(in_channels=in_channels,n_filters=n_filters,
                 res_drop=res_drop,attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop,)
        self.encphi = Encoder5(in_channels=in_channels,n_filters=n_filters,
                 res_drop=res_drop,attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop,)
        self.dec = Decoder5(out_channels=out_channels,n_filters=n_filters,res_drop=res_drop)
        self.kno1 = KNOBlock(c=n_filters,h=img_size,w=img_size)
        self.kno2 = KNOBlock(c=2*n_filters,h=img_size/2,w=img_size/2)
        self.kno3 = KNOBlock(c=4*n_filters,h=img_size/4,w=img_size/4)
        self.kno4 = KNOBlock(c=8*n_filters,h=img_size/8,w=img_size/8)
        self.kno5 = KNOBlock(c=16*n_filters,h=img_size/16,w=img_size/16)

        self.apply(self._init_weights)
        self.kno1.kno._init_weights()
        self.kno2.kno._init_weights()
        self.kno3.kno._init_weights()
        self.kno4.kno._init_weights()
        self.kno5.kno._init_weights()

    def forward(self,x,p=1):
        r11,r23,r32,r42,r52 = self.encr(x)
        phi11,phi23,phi32,phi42,phi52 = self.encphi(x)
        phi11 = self.kno1(phi11,p=p)
        phi23 = self.kno2(phi23,p=p)
        phi32 = self.kno3(phi32,p=p)
        phi42 = self.kno4(phi42,p=p)
        phi52 = self.kno5(phi52,p=p)
        x11 = r11 * torch.cos(phi11)
        x23 = r23 * torch.cos(phi23)
        x32 = r32 * torch.cos(phi32)
        x42 = r42 * torch.cos(phi42)
        x52 = r52 * torch.cos(phi52)
        return self.dec(x11,x23,x32,x42,x52)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

class Encoder4(nn.Module):
    def __init__(self,in_channels=4,n_filters=4,
                 res_drop=0.5,attn_drop=0.5,proj_drop=0.5,ffn_drop=0.5,):
        super().__init__()
        self.block11=ResidualBlock(in_chans=in_channels, out_chans=n_filters,
                            stride=1, groups=1, drop=res_drop)
        self.block21=BasicConv2d(in_chans=in_channels,out_chans=in_channels,
                            kernel_size=3, stride=1, padding=1)
        self.block22=ResidualBlock(in_chans=in_channels, out_chans=2*n_filters,
                            stride=2, groups=1, drop=res_drop)
        self.block31=FEBlock(in_chans=2*n_filters,features=4*n_filters,
                            patch_size=3,stride=2,growth_rate=4*n_filters,
                            attn_depth=4,ffn_ratio=2,dct_depth=2,
                            num_heads=8,d_model=8*n_filters,
                            attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop
                            )
        self.block41=FEBlock(in_chans=4*n_filters,features=8*n_filters,
                            patch_size=3,stride=2,growth_rate=4*n_filters,
                            attn_depth=4,ffn_ratio=2,dct_depth=2,
                            num_heads=8,d_model=8*n_filters,
                            attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop
                            )

        self.down12=nn.Conv2d(n_filters,2*n_filters,kernel_size=3,stride=2,padding=1)
        self.down23=nn.Conv2d(2*n_filters,4*n_filters,kernel_size=3,stride=2,padding=1)
        self.down34=nn.Conv2d(4*n_filters,8*n_filters,kernel_size=3,stride=2,padding=1)

        self.block23=BasicConv2d(in_chans=2*n_filters,out_chans=2*n_filters,kernel_size=3,stride=1,padding=1)
        self.block32=BasicConv2d(in_chans=4*n_filters,out_chans=4*n_filters,kernel_size=3,stride=1,padding=1)
        self.block42=BasicConv2d(in_chans=8*n_filters,out_chans=8*n_filters,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        # x11
        x11 = self.block11(x)

        # x23
        x21 = self.block21(x)
        x22 = self.block22(x21)
        x23 = self.block23(self.down12(x11)+x22)

        # x32
        x31 = self.block31(x22)
        x32 = self.block32(x31+self.down23(x23))

        # x42
        x41 = self.block41(x31)
        x42 = self.block42(x41+self.down34(x32))

        return [x11,x23,x32,x42]


class Decoder4(nn.Module):
    def __init__(self,out_channels=1,n_filters=4,res_drop=0.5):
        super().__init__()
        self.up43=nn.ConvTranspose2d(8*n_filters,4*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.up32=nn.ConvTranspose2d(4*n_filters,2*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.up21=nn.ConvTranspose2d(2*n_filters,1*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)

        self.block33=BasicConv2d(in_chans=8*n_filters,out_chans=8*n_filters,kernel_size=3,stride=1,padding=1)
        self.block34=ResidualBlock(in_chans=8*n_filters, out_chans=4*n_filters,
                            stride=1, groups=1, drop=res_drop)

        self.block24=BasicConv2d(in_chans=4*n_filters,out_chans=4*n_filters,kernel_size=3,stride=1,padding=1)
        self.block25=ResidualBlock(in_chans=4*n_filters, out_chans=2*n_filters,
                            stride=1, groups=1, drop=res_drop)

        self.block12=BasicConv2d(in_chans=2*n_filters,out_chans=2*n_filters,kernel_size=3,stride=1,padding=1)
        self.block13=nn.Conv2d(in_channels=2*n_filters,out_channels=out_channels,kernel_size=1,stride=1,padding=0) # 1*1的卷积核


    def forward(self,x11,x23,x32,x42):
        x33 = self.block33(torch.cat([x32,self.up43(x42)],dim=1))
        x34 = self.block34(x33)
        x24 = self.block24(torch.cat([x23,self.up32(x34)],dim=1))
        x25 = self.block25(x24)
        x12 = self.block12(torch.cat([x11,self.up21(x25)],dim=1))
        x13 = self.block13(x12)
        return x13

class MyModel4(nn.Module):
    def __init__(self,img_size=128,in_channels=3,out_channels=1,n_filters=16,
                 res_drop=0.5,attn_drop=0.5,proj_drop=0.5,ffn_drop=0.5,):
        super().__init__()
        self.encr = Encoder4(in_channels=in_channels,n_filters=n_filters,
                 res_drop=res_drop,attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop,)
        self.encphi = Encoder4(in_channels=in_channels,n_filters=n_filters,
                 res_drop=res_drop,attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop,)
        self.dec = Decoder4(out_channels=out_channels,n_filters=n_filters,res_drop=res_drop)
        self.kno1 = KNOBlock(c=n_filters,h=img_size,w=img_size)
        self.kno2 = KNOBlock(c=2*n_filters,h=img_size/2,w=img_size/2)
        self.kno3 = KNOBlock(c=4*n_filters,h=img_size/4,w=img_size/4)
        self.kno4 = KNOBlock(c=8*n_filters,h=img_size/8,w=img_size/8)

        self.apply(self._init_weights)
        self.kno1.kno._init_weights()
        self.kno2.kno._init_weights()
        self.kno3.kno._init_weights()
        self.kno4.kno._init_weights()

    def forward(self,x,p=1):
        r11,r23,r32,r42 = self.encr(x)
        phi11,phi23,phi32,phi42 = self.encphi(x)
        phi11 = self.kno1(phi11,p=p)
        phi23 = self.kno2(phi23,p=p)
        phi32 = self.kno3(phi32,p=p)
        phi42 = self.kno4(phi42,p=p)
        x11 = r11 * torch.cos(phi11)
        x23 = r23 * torch.cos(phi23)
        x32 = r32 * torch.cos(phi32)
        x42 = r42 * torch.cos(phi42)
        return self.dec(x11,x23,x32,x42)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
class Encoder3(nn.Module):
    def __init__(self,in_channels=4,n_filters=4,
                 res_drop=0.5,attn_drop=0.5,proj_drop=0.5,ffn_drop=0.5,):
        super().__init__()
        self.block11=ResidualBlock(in_chans=in_channels, out_chans=n_filters,
                            stride=1, groups=1, drop=res_drop)
        self.block21=BasicConv2d(in_chans=in_channels,out_chans=in_channels,
                            kernel_size=3, stride=1, padding=1)
        self.block22=ResidualBlock(in_chans=in_channels, out_chans=2*n_filters,
                            stride=2, groups=1, drop=res_drop)
        self.block31=FEBlock(in_chans=2*n_filters,features=4*n_filters,
                            patch_size=3,stride=2,growth_rate=4*n_filters,
                            attn_depth=4,ffn_ratio=2,dct_depth=2,
                            num_heads=8,d_model=8*n_filters,
                            attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop
                            )

        self.down12=nn.Conv2d(n_filters,2*n_filters,kernel_size=3,stride=2,padding=1)
        self.down23=nn.Conv2d(2*n_filters,4*n_filters,kernel_size=3,stride=2,padding=1)

        self.block23=BasicConv2d(in_chans=2*n_filters,out_chans=2*n_filters,kernel_size=3,stride=1,padding=1)
        self.block32=BasicConv2d(in_chans=4*n_filters,out_chans=4*n_filters,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        # x11
        x11 = self.block11(x)

        # x23
        x21 = self.block21(x)
        x22 = self.block22(x21)
        x23 = self.block23(self.down12(x11)+x22)

        # x32
        x31 = self.block31(x22)
        x32 = self.block32(x31+self.down23(x23))

        return [x11,x23,x32]


class Decoder3(nn.Module):
    def __init__(self,out_channels=1,n_filters=4,res_drop=0.5):
        super().__init__()
        self.up32=nn.ConvTranspose2d(4*n_filters,2*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.up21=nn.ConvTranspose2d(2*n_filters,1*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)

        self.block24=BasicConv2d(in_chans=4*n_filters,out_chans=4*n_filters,kernel_size=3,stride=1,padding=1)
        self.block25=ResidualBlock(in_chans=4*n_filters, out_chans=2*n_filters,
                            stride=1, groups=1, drop=res_drop)

        self.block12=BasicConv2d(in_chans=2*n_filters,out_chans=2*n_filters,kernel_size=3,stride=1,padding=1)
        self.block13=nn.Conv2d(in_channels=2*n_filters,out_channels=out_channels,kernel_size=1,stride=1,padding=0) # 1*1的卷积核


    def forward(self,x11,x23,x32):
        x24 = self.block24(torch.cat([x23,self.up32(x32)],dim=1))
        x25 = self.block25(x24)
        x12 = self.block12(torch.cat([x11,self.up21(x25)],dim=1))
        x13 = self.block13(x12)
        return x13

class MyModel3(nn.Module):
    def __init__(self,img_size=128,in_channels=3,out_channels=1,n_filters=16,
                 res_drop=0.5,attn_drop=0.5,proj_drop=0.5,ffn_drop=0.5,):
        super().__init__()
        self.encr = Encoder3(in_channels=in_channels,n_filters=n_filters,
                 res_drop=res_drop,attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop,)
        self.encphi = Encoder3(in_channels=in_channels,n_filters=n_filters,
                 res_drop=res_drop,attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop,)
        self.dec = Decoder3(out_channels=out_channels,n_filters=n_filters,res_drop=res_drop)
        self.kno1 = KNOBlock(c=n_filters,h=img_size,w=img_size)
        self.kno2 = KNOBlock(c=2*n_filters,h=img_size/2,w=img_size/2)
        self.kno3 = KNOBlock(c=4*n_filters,h=img_size/4,w=img_size/4)

        self.apply(self._init_weights)
        self.kno1.kno._init_weights()
        self.kno2.kno._init_weights()
        self.kno3.kno._init_weights()

    def forward(self,x,p=1):
        r11,r23,r32 = self.encr(x)
        phi11,phi23,phi32 = self.encphi(x)
        phi11 = self.kno1(phi11,p=p)
        phi23 = self.kno2(phi23,p=p)
        phi32 = self.kno3(phi32,p=p)
        x11 = r11 * torch.cos(phi11)
        x23 = r23 * torch.cos(phi23)
        x32 = r32 * torch.cos(phi32)
        return self.dec(x11,x23,x32)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
            

def getpath(case,result='result1'):
    '''
    Args:
        case (int): 0: shear flow, 1: Kolmogorov flow, 2: Gray-Scott reaction-diffusion
        result(str) : e.g. 'result1' -> pytorch_model.bin,train.log,ds_config.json
    Returns:
        data_path (str): path to the test data
        model_path (str): path to the trained model
    '''    
    if case == 0:
        json_path = '../shear/'+ result + '/ds_config.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f) # 读取配置文件 df_config.json
        data_path = config["test_data_path"]
        data_path = data_path.replace("./test", "../shear/data/test")
        model_path = '../shear/'+ result + '/pytorch_model.bin'
        n_filters = config['n_filters']
        log_path = '../shear/'+ result + '/train.log'
    elif case == 1:
        json_path = '../kol/'+ result + '/ds_config.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f) # 读取配置文件 df_config.json
        data_path = config["test_data_path"]
        data_path = data_path.replace("./data", "../kol/data")
        model_path = '../kol/'+ result + '/pytorch_model.bin'
        n_filters = config['n_filters']  
        log_path = '../kol/'+ result + '/train.log' 
    elif case == 2:    
        json_path = '../gray/'+ result + '/ds_config.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f) # 读取配置文件 df_config.json
        data_path = config["test_data_path"]
        data_path = data_path.replace("./test", "../gray/data/test")
        model_path = '../gray/'+ result + '/pytorch_model.bin'
        n_filters = config['n_filters']   
        log_path = '../gray/'+ result + '/train.log'       
             
    return data_path,model_path,n_filters,log_path


def getdata(data_path,case):
    '''
    Args:
        data_path (str): path
        case (int): 0: shear flow, 1: Kolmogorov flow, 2: Gray-Scott
    Returns:
        hint,vort np.array (n,t,h,h) (n,2,h,h)
    '''    
    vort_list,u0_list,v0_list = [],[],[]
    for path in data_path.split():
        with h5py.File(path,'r') as f:
            if case ==0:
                vort_list.append(f['vort'][:,:,:,:].astype(np.float32))
                u0_list.append(f['u0'][:,:,:,:].astype(np.float32))
                v0_list.append(f['v0'][:,:,:,:].astype(np.float32))
            elif case ==1:
                vort_list.append(f['vort'][:240,:,:,:].astype(np.float32))
                u0_list.append(f['u0'][:240,:,:,:].astype(np.float32))
                v0_list.append(f['v0'][:240,:,:,:].astype(np.float32))
            elif case ==2:
                vort_list.append(f['A'][:,:,:,:].astype(np.float32))
                u0_list.append(f['A0'][:,:,:,:].astype(np.float32))
                v0_list.append(f['B0'][:,:,:,:].astype(np.float32))                
    vort = np.concatenate(vort_list,axis=0) # (n,t,h,h)
    u0 = np.concatenate(u0_list,axis=0) # (n,1,h,h)
    v0 = np.concatenate(v0_list,axis=0) # (n,1,h,h)
    hint = np.concatenate([u0,v0],axis=1) # (n,2,h,h)
    return hint,vort

def getcase(case,result='result1'):
    '''
    Args:
        case (int): 0: shear flow, 1: Kolmogorov flow, 2: Gray-Scott
        result(str) : e.g. 'result1' -> pytorch_model.bin,train.log,ds_config.json
    Returns:
        hint,vort np.array (n,t,h,h) (n,2,h,h)
        net (MyModel): trained model
    ''' 
    data_path,model_path,n_filters,log_path = getpath(case=case,result=result)
    hint,vort = getdata(data_path=data_path,case=case)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mycheck = torch.load(model_path,map_location=device)
    if result.startswith('result_5'):
        net = MyModel5(n_filters=n_filters).to(device)
    elif result.startswith('result_4'):
        net = MyModel4(n_filters=n_filters).to(device)
    elif result.startswith('result_3'):
        net = MyModel3(n_filters=n_filters).to(device)
    else:
        net = MyModel5(n_filters=n_filters).to(device)        
    net.load_state_dict(mycheck)
    
    # 定义训练日志和验证日志的正则表达式模式
    train_pattern = re.compile(
        r'\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]\s\[INFO\]\sEpoch:\s*(?P<epoch>\d+)\s*\|\s*Time:\s*(?P<time>[\d.]+)s\s*\|\s*Train Loss:\s*(?P<train_loss>[\d.e+-]+)\s*\|\s*LR:\s*(?P<lr>[\d.e+-]+)'
    )
    val_pattern = re.compile(
        r'\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]\s\[INFO\]\sEpoch:\s*(?P<epoch>\d+)\s*\|\s*Time:\s*(?P<time>[\d.]+)s\s*\|\s*Val\s+Loss:\s*(?P<val_loss>[\d.e+-]+)'
    )
    # 存储解析结果的字典
    log_dict = {
        'train':{
            'epoch':[],
            'time': [],
            'loss': [],  
            'lr':[],          
        },
        'val':{
            'epoch':[],
            'time': [],
            'loss': [],  
        }
    }
    
    with open(log_path, 'r') as file:
        for _, line in enumerate(file, 1):
            # 尝试匹配训练日志
            train_match = train_pattern.search(line)
            if train_match:
                epoch = int(train_match.group('epoch'))
                log_dict['train']['epoch'].append(epoch)
                log_dict['train']['time'].append(float(train_match.group('time')))
                log_dict['train']['loss'].append(float(train_match.group('train_loss')))
                log_dict['train']['lr'].append(float(train_match.group('lr')))
                continue
                
            # 尝试匹配验证日志
            val_match = val_pattern.search(line)
            if val_match:
                epoch = int(val_match.group('epoch'))
                log_dict['val']['epoch'].append(epoch)
                log_dict['val']['time'].append(float(val_match.group('time')))
                log_dict['val']['loss'].append(float(val_match.group('val_loss')))

    return hint,vort,net,log_dict

def measure_inference_memory(model,batch_size=1, input_shape=(3,128,128)):
    """计算不同 Batch Size 下的推理内存"""
    model.eval().cuda()
    print(f"初始显存占用: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"模型显存: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3):.2f} GB")    
    torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
    torch.cuda.empty_cache()  # 清空缓存
    mem_before = torch.cuda.memory_allocated()  # 初始显存

    inputs = torch.randn(batch_size, *input_shape).cuda()
    p = torch.tensor(0).unsqueeze(-1).cuda()

    with torch.no_grad():
        _ = model(inputs, p=p)

    mem_after = torch.cuda.max_memory_allocated()  # 峰值显存
    return (mem_after - mem_before) / (1024 ** 2)  # MB

def work(kwargs):
    '''
    Args:
        hint (b,2,h,h)
        vort (b,t,h,h)
        net (b,3,h,h)->(b,1,h,h)
    Returns:
        v1: np.array (b,t,h,h) GT
        v3: np.array (b,t,h,h) QKM
    '''    
    id = kwargs['id']
    case = kwargs['case']
    result = kwargs['result']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hint, vort, net, _ = getcase(case=case, result=result)
    hint = torch.from_numpy(hint.astype(np.float32)).to(device)
    v1 = torch.from_numpy(vort.astype(np.float32)).to(device)
    h = hint.shape[-1]
    net.eval()
    with torch.no_grad():
        # QKM
        v3 = torch.zeros(v1.shape[0],v1.shape[1],h,h).to(device)
        input3=torch.zeros(hint.shape[0],3,h,h).to(device)
        input3[:,:2,:,:]=hint[:,:2,:,:]
        input3[:,2,:,:]=v1[:,0,:,:]

        for i in range(v1.shape[1]):        
            p = torch.tensor(i).unsqueeze(-1).to(device)
            v3[:,i,:,:] = net(input3,p=p).squeeze()
            print(f'[id={id}] [case={case:d}] [{result:s}] [k={i:d}] Done!')
    
    v1=v1.cpu()
    v3=v3.cpu()
    with torch.no_grad():
        v1 = v1.detach().numpy()
        v3 = v3.detach().numpy()
    np.save(f'v1_case={case:d}_{result:s}.npy', v1)
    np.save(f'v3_case={case:d}_{result:s}.npy', v3)
    
    for bs in [1,240]:
        memory = measure_inference_memory(net, batch_size=bs)
        print(f"[id={id}] [case={case:d}] [{result:s}] Batch Size={bs} → 内存占用: {memory:.2f} MB")
    del hint, v1
    torch.cuda.empty_cache()
    
    return id
    

if __name__ == '__main__':
    total_tstart = time.time()
    
    for idx, (case, result) in enumerate([(c, r) for c in [1,2,0] 
                                         for r in ['result_5_16','result_5_12','result_5_8',
                                                 'result_4_16','result_4_12','result_4_8',
                                                 'result_3_16','result_3_12','result_3_8']]):
        work({'id': idx, 'case': case, 'result': result})
        
    for _, (case, result) in enumerate([(c, r) for c in [1,2,0] 
                                            for r in ['result_5_16','result_5_12','result_5_8',
                                                    'result_4_16','result_4_12','result_4_8',
                                                    'result_3_16','result_3_12','result_3_8']]):
        v1_list = np.load(f'v1_case={case:d}_{result:s}.npy')
        v3_list = np.load(f'v3_case={case:d}_{result:s}.npy')
        mse = ( (v3_list-v1_list)**2 ).mean(axis=(-2,-1))
        relative_mse = ( mse / (v1_list**2).mean(axis=(-2,-1)) )
        np.save(f'RMSE_case={case:d}_{result:s}.npy', relative_mse)
        print(f'[case={case:d}] [{result:s}] [RMSE={relative_mse.mean():.4e}]')
        
    total_tend = time.time()
    print(f"\n总耗时: {(total_tend - total_tstart):.2f} [s]")