# -*- coding: utf-8 -*-
import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import shutil
import argparse
from torch.utils.checkpoint import checkpoint
from deepspeed import get_accelerator
from timeit import default_timer
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import deepspeed
from torch.profiler import profile, ProfilerActivity
import json
import logging
import torch.distributed as dist
from timm.layers import trunc_normal_

import glob
import re
import gc
from tqdm import tqdm
from collections import OrderedDict
from dataclasses import dataclass
from deepspeed.utils import logger
from deepspeed.checkpoint.constants import (DS_VERSION, OPTIMIZER_STATE_DICT, SINGLE_PARTITION_OF_FP32_GROUPS,
                                            FP32_FLAT_GROUPS, ZERO_STAGE, PARTITION_COUNT, PARAM_SHAPES, BUFFER_NAMES,
                                            FROZEN_PARAM_SHAPES, FROZEN_PARAM_FRAGMENTS)

'''
chechpoint: DCTBlock, FEBlock
torch 2.1.0
deepspeed 0.16.5
'''

class MyDataset(Dataset):

    def __init__(self, datapath):
        vort_list = []
        u0_list = []
        v0_list = []
        for path in datapath.split():
            with h5py.File(path,'r') as f:
                vort_list.append(torch.from_numpy(f['A'][:,:,:,:].astype(np.float32)))
                u0_list.append(torch.from_numpy(f['A0'][:,:,:,:].astype(np.float32)))
                v0_list.append(torch.from_numpy(f['B0'][:,:,:,:].astype(np.float32)))

        vort = torch.cat(vort_list,dim=0) # (c, t, h, h)
        u0 = torch.cat(u0_list,dim=0) # (c, 1, h, h)
        v0 = torch.cat(v0_list,dim=0) # (c, 1, h, h)

        c=vort.shape[0]
        t=vort.shape[1]
        h=vort.shape[2]

        k = 2*t-1
        n = c * k

        self.inputs=torch.zeros((n,3,h,h))
        self.labels=torch.zeros((n))
        self.targets=torch.zeros((n,1,h,h))

        for i in range(c):
            self.inputs[i*k:(i+1)*k,0,:,:]=u0[i,0,:,:]
            self.inputs[i*k:(i+1)*k,1,:,:]=v0[i,0,:,:]

            self.inputs[i*k:i*k+t,2,:,:]=vort[i,0,:,:] # p=arange(t)
            self.labels[i*k:i*k+t]=torch.arange(t)
            self.targets[i*k:i*k+t,0,:,:]=vort[i,:,:,:]

            self.inputs[i*k+t:(i+1)*k,2,:,:]=vort[i,1:,:,:] # AE
            self.labels[i*k+t:(i+1)*k]=0.0
            self.targets[i*k+t:(i+1)*k,0,:,:]=vort[i,1:,:,:]

        self.labels = self.labels.unsqueeze(-1).unsqueeze(-1)


    def __len__(self):
        return self.inputs.shape[0] #  n
    def __getitem__(self, idx):

        return self.inputs[idx,:,:,:],self.labels[idx],self.targets[idx,:,:,:]


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): (batch_size, time_step, feature1_size, feature2_size)
        '''
        bs = x.shape[0]
        return F.mse_loss(x.view(bs,-1),target.view(bs,-1),reduction='mean') / ((target**2).mean() + 1.e-5) # eps=1e-5




class ResidualBlock(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1, groups=1, act_layer=nn.SiLU, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=True) # 3x3
        self.bn1 = nn.BatchNorm2d(out_chans)

        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True)  # 3x3
        self.bn2 = nn.BatchNorm2d(out_chans)

        self.act = act_layer()  #
        self.dropout = nn.Dropout(drop)  # Dropout


        self.downsample = None
        if stride != 1 or in_chans != out_chans:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x
        out = self.dropout(out)
        out = self.act(out)  #
        return out


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim) #

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class MixFFN(nn.Module):
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
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class AttentionBlock(nn.Module):
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
        return checkpoint(self._forward_src,x,use_reentrant=False)

    def _forward_src(self,x):
        B = x.size(0) # batch_size
        x,H,W = self.embed(x)
        for layer in self.dctblocks:
            x = layer(x,H=H,W=W)
        return x.view(B,-1,H,W)


class BasicConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, **kwargs),
            nn.BatchNorm2d(out_chans),
            nn.SiLU(inplace=True) # inplace=True
        )

    def forward(self, x):
        return self.conv(x)


class UnitBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = int(features)
        self.weight_len = int(torch.log2(torch.tensor(features)))
        self.weight = nn.Parameter(torch.zeros(self.weight_len))  # 0
        self.register_buffer(
            "binary_bases",
            2 ** (self.weight_len - 1 - torch.arange(self.weight_len)),
            persistent=False
        )
    def forward(self, input_phi, p=torch.tensor(1.)):
        z = self._generate_value_optimized()
        return input_phi + p.view(-1,1) * z.view(1,-1)  #
    def _generate_value_optimized(self):
        binary_mask = (torch.arange(self.features, device=self.weight.device).unsqueeze(1) // self.binary_bases) % 2
        z = torch.sum(binary_mask * self.weight, dim=1) - torch.sum(self.weight) / 2
        return z
    def _init_weights(self):
        nn.init.constant_(self.weight, 0.0)  # 0


class KNOBlock(nn.Module):
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

class Encoder(nn.Module):
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

        x11 = self.block11(x)


        x21 = self.block21(x)
        x22 = self.block22(x21)
        x23 = self.block23(self.down12(x11)+x22)


        x31 = self.block31(x22)
        x32 = self.block32(x31+self.down23(x23))

        return [x11,x23,x32]


class Decoder(nn.Module):
    def __init__(self,out_channels=1,n_filters=4,res_drop=0.5):
        super().__init__()
        self.up32=nn.ConvTranspose2d(4*n_filters,2*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.up21=nn.ConvTranspose2d(2*n_filters,1*n_filters,kernel_size=3,stride=2,padding=1,output_padding=1)

        self.block24=BasicConv2d(in_chans=4*n_filters,out_chans=4*n_filters,kernel_size=3,stride=1,padding=1)
        self.block25=ResidualBlock(in_chans=4*n_filters, out_chans=2*n_filters,
                            stride=1, groups=1, drop=res_drop)

        self.block12=BasicConv2d(in_chans=2*n_filters,out_chans=2*n_filters,kernel_size=3,stride=1,padding=1)
        self.block13=nn.Conv2d(in_channels=2*n_filters,out_channels=out_channels,kernel_size=1,stride=1,padding=0) # 1*1


    def forward(self,x11,x23,x32):
        x24 = self.block24(torch.cat([x23,self.up32(x32)],dim=1))
        x25 = self.block25(x24)
        x12 = self.block12(torch.cat([x11,self.up21(x25)],dim=1))
        x13 = self.block13(x12)
        return x13

class MyModel(nn.Module):
    def __init__(self,img_size=128,in_channels=3,out_channels=1,n_filters=16,
                 res_drop=0.5,attn_drop=0.5,proj_drop=0.5,ffn_drop=0.5,):
        super().__init__()
        self.encr = Encoder(in_channels=in_channels,n_filters=n_filters,
                 res_drop=res_drop,attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop,)
        self.encphi = Encoder(in_channels=in_channels,n_filters=n_filters,
                 res_drop=res_drop,attn_drop=attn_drop,proj_drop=proj_drop,ffn_drop=ffn_drop,)
        self.dec = Decoder(out_channels=out_channels,n_filters=n_filters,res_drop=res_drop)
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

''' DeepSpeed Team

This script extracts fp32 consolidated weights from a zero 1, 2 and 3 DeepSpeed checkpoints. It gets
copied into the top level checkpoint dir, so the user can easily do the conversion at any point in
the future. Once extracted, the weights don't require DeepSpeed and can be used in any
application.

example:    python zero_to_fp32.py . output_dir/
'''

@dataclass
class zero_model_state:
    buffers: dict
    param_shapes: dict
    shared_params: list
    ds_version: int
    frozen_param_shapes: dict
    frozen_param_fragments: dict

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_model_state_file(checkpoint_dir, zero_stage):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")


    if zero_stage <= 2:
        file = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    elif zero_stage == 3:
        file = os.path.join(checkpoint_dir, "zero_pp_rank_0_mp_rank_00_model_states.pt")

    if not os.path.exists(file):
        raise FileNotFoundError(f"can't find model states file at '{file}'")

    return file


def get_checkpoint_files(checkpoint_dir, glob_pattern):

    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern)), key=natural_keys)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"can't find {glob_pattern} files in directory '{checkpoint_dir}'")

    return ckpt_files


def get_optim_files(checkpoint_dir):
    return get_checkpoint_files(checkpoint_dir, "*_optim_states.pt")


def get_model_state_files(checkpoint_dir):
    return get_checkpoint_files(checkpoint_dir, "*_model_states.pt")


def parse_model_states(files):
    zero_model_states = []
    for file in files:
        state_dict = torch.load(file, map_location="cpu", weights_only=False)

        if BUFFER_NAMES not in state_dict:
            raise ValueError(f"{file} is not a model state checkpoint")
        buffer_names = state_dict[BUFFER_NAMES]



        buffers = {k: v.float() for k, v in state_dict["module"].items() if k in buffer_names}
        param_shapes = state_dict[PARAM_SHAPES]


        param_names = []
        for s in param_shapes:
            for name in s.keys():
                param_names.append(name)


        frozen_param_shapes = state_dict.get(FROZEN_PARAM_SHAPES, None)
        if frozen_param_shapes is not None:
            param_names += list(frozen_param_shapes.keys())


        shared_params = [[k, v] for k, v in state_dict["shared_params"].items()]

        ds_version = state_dict.get(DS_VERSION, None)

        frozen_param_fragments = state_dict.get(FROZEN_PARAM_FRAGMENTS, None)

        z_model_state = zero_model_state(buffers=buffers,
                                         param_shapes=param_shapes,
                                         shared_params=shared_params,
                                         ds_version=ds_version,
                                         frozen_param_shapes=frozen_param_shapes,
                                         frozen_param_fragments=frozen_param_fragments)
        zero_model_states.append(z_model_state)

    return zero_model_states


def parse_optim_states(files, ds_checkpoint_dir):
    total_files = len(files)
    state_dicts = []
    for f in tqdm(files, desc='Loading checkpoint shards'):
        state_dict = torch.load(f, map_location="cpu", mmap=True, weights_only=False)


        state_dict["optimizer_state_dict"].pop("optimizer_state_dict", None)
        state_dicts.append(state_dict)

    if not ZERO_STAGE in state_dicts[0][OPTIMIZER_STATE_DICT]:
        raise ValueError(f"{files[0]} is not a zero checkpoint")
    zero_stage = state_dicts[0][OPTIMIZER_STATE_DICT][ZERO_STAGE]
    world_size = state_dicts[0][OPTIMIZER_STATE_DICT][PARTITION_COUNT]





    if type(world_size) is list:
        world_size = max(world_size)

    if world_size != total_files:
        raise ValueError(
            f"Expected {world_size} of '*_optim_states.pt' under '{ds_checkpoint_dir}' but found {total_files} files. "
            "Possibly due to an overwrite of an old checkpoint, or a checkpoint didn't get saved by one or more processes."
        )


    if zero_stage <= 2:
        fp32_groups_key = SINGLE_PARTITION_OF_FP32_GROUPS
    elif zero_stage == 3:
        fp32_groups_key = FP32_FLAT_GROUPS
    else:
        raise ValueError(f"unknown zero stage {zero_stage}")

    fp32_flat_groups = [state_dicts[i][OPTIMIZER_STATE_DICT][fp32_groups_key] for i in range(len(state_dicts))]
    return zero_stage, world_size, fp32_flat_groups


def _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir, exclude_frozen_parameters):
    """
    Returns fp32 state_dict reconstructed from ds checkpoint

    Args:
        - ``ds_checkpoint_dir``: path to the deepspeed checkpoint folder (where the optimizer files are)

    """
    print(f"Processing zero checkpoint '{ds_checkpoint_dir}'")

    optim_files = get_optim_files(ds_checkpoint_dir)
    zero_stage, world_size, fp32_flat_groups = parse_optim_states(optim_files, ds_checkpoint_dir)
    print(f"Detected checkpoint of type zero stage {zero_stage}, world_size: {world_size}")

    model_files = get_model_state_files(ds_checkpoint_dir)

    zero_model_states = parse_model_states(model_files)
    print(f'Parsing checkpoint created by deepspeed=={zero_model_states[0].ds_version}')

    if zero_stage <= 2:
        return _get_fp32_state_dict_from_zero2_checkpoint(world_size, fp32_flat_groups, zero_model_states,
                                                          exclude_frozen_parameters)
    elif zero_stage == 3:
        return _get_fp32_state_dict_from_zero3_checkpoint(world_size, fp32_flat_groups, zero_model_states,
                                                          exclude_frozen_parameters)


def _zero2_merge_frozen_params(state_dict, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(zero_model_states[0].frozen_param_shapes) == 0:
        return

    frozen_param_shapes = zero_model_states[0].frozen_param_shapes
    frozen_param_fragments = zero_model_states[0].frozen_param_fragments

    total_params = 0
    total_numel = 0
    for name, shape in frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        state_dict[name] = frozen_param_fragments[name]

    print(f"Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements")


def _has_callable(obj, fn):
    attr = getattr(obj, fn, None)
    return callable(attr)


def _zero2_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states):
    param_shapes = zero_model_states[0].param_shapes







    num_param_groups = len(fp32_flat_groups[0])
    merged_single_partition_of_fp32_groups = []
    for i in range(num_param_groups):
        merged_partitions = [sd[i] for sd in fp32_flat_groups]
        full_single_fp32_vector = torch.cat(merged_partitions, 0)
        merged_single_partition_of_fp32_groups.append(full_single_fp32_vector)
    avail_numel = sum(
        [full_single_fp32_vector.numel() for full_single_fp32_vector in merged_single_partition_of_fp32_groups])




    total_numel = 0
    total_params = 0
    for shapes, full_single_fp32_vector in zip(param_shapes, merged_single_partition_of_fp32_groups):
        offset = 0
        avail_numel = full_single_fp32_vector.numel()
        for name, shape in shapes.items():

            unpartitioned_numel = shape.numel() if _has_callable(shape, 'numel') else math.prod(shape)
            total_numel += unpartitioned_numel
            total_params += 1

            state_dict[name] = full_single_fp32_vector.narrow(0, offset, unpartitioned_numel).view(shape)
            offset += unpartitioned_numel





        align_to = 2 * world_size

        def zero2_align(x):
            return align_to * math.ceil(x / align_to)


        offset = zero2_align(offset)
        avail_numel = zero2_align(avail_numel)



        if offset != avail_numel:
            raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")

    print(f"Reconstructed fp32 state dict with {total_params} params {total_numel} elements")


def _get_fp32_state_dict_from_zero2_checkpoint(world_size, fp32_flat_groups, zero_model_states,
                                               exclude_frozen_parameters):
    state_dict = OrderedDict()


    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)

    if not exclude_frozen_parameters:
        _zero2_merge_frozen_params(state_dict, zero_model_states)

    _zero2_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states)


    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]

    return state_dict


def zero3_partitioned_param_info(unpartitioned_numel, world_size):
    remainder = unpartitioned_numel % world_size
    padding_numel = (world_size - remainder) if remainder else 0
    partitioned_numel = math.ceil(unpartitioned_numel / world_size)
    return partitioned_numel, padding_numel


def _zero3_merge_frozen_params(state_dict, world_size, zero_model_states):
    if zero_model_states[0].frozen_param_shapes is None or len(zero_model_states[0].frozen_param_shapes) == 0:
        return

    total_params = 0
    total_numel = 0
    for name, shape in zero_model_states[0].frozen_param_shapes.items():
        total_params += 1
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel

        param_frags = tuple(model_state.frozen_param_fragments[name] for model_state in zero_model_states)
        state_dict[name] = torch.cat(param_frags, 0).narrow(0, 0, unpartitioned_numel).view(shape)

        partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)

    print(f"Reconstructed Frozen fp32 state dict with {total_params} params {total_numel} elements")


class GatheredTensor:
    """
    A pseudo tensor that collects partitioned weights.
    It is more memory efficient when there are multiple groups.
    """

    def __init__(self, flat_groups, flat_groups_offset, offset, partitioned_numel, shape):
        self.flat_groups = flat_groups
        self.flat_groups_offset = flat_groups_offset
        self.offset = offset
        self.partitioned_numel = partitioned_numel
        self.shape = shape
        self.dtype = self.flat_groups[0][0].dtype

    def contiguous(self):
        """
        Merge partitioned weights from flat_groups into a single tensor.
        """
        end_idx = self.offset + self.partitioned_numel
        world_size = len(self.flat_groups)
        pad_flat_param_chunks = []

        for rank_i in range(world_size):

            flat_groups_at_rank_i = self.flat_groups[rank_i]
            start_group_id = None
            end_group_id = None
            for group_id in range(len(self.flat_groups_offset)):
                if self.flat_groups_offset[group_id] <= self.offset < self.flat_groups_offset[group_id + 1]:
                    start_group_id = group_id
                if self.flat_groups_offset[group_id] < end_idx <= self.flat_groups_offset[group_id + 1]:
                    end_group_id = group_id
                    break

            for group_id in range(start_group_id, end_group_id + 1):
                flat_tensor = flat_groups_at_rank_i[group_id]
                start_offset = self.offset - self.flat_groups_offset[group_id]
                end_offset = min(end_idx, self.flat_groups_offset[group_id + 1]) - self.flat_groups_offset[group_id]
                pad_flat_param_chunks.append(flat_tensor[start_offset:end_offset])


        pad_flat_param = torch.cat(pad_flat_param_chunks, dim=0)
        param = pad_flat_param[:self.shape.numel()].view(self.shape).contiguous()
        return param


def _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states):
    param_shapes = zero_model_states[0].param_shapes
    avail_numel = sum([flat_group.numel() for flat_group in fp32_flat_groups[0]]) * world_size





    param_shapes = {k: v for d in param_shapes for k, v in d.items()}




    offset = 0
    total_numel = 0
    total_params = 0
    flat_groups_offset = [0] + list(np.cumsum([flat_tensor.numel() for flat_tensor in fp32_flat_groups[0]]))
    for name, shape in tqdm(param_shapes.items(), desc='Gathering sharded weights'):
        unpartitioned_numel = shape.numel()
        total_numel += unpartitioned_numel
        total_params += 1
        partitioned_numel, partitioned_padding_numel = zero3_partitioned_param_info(unpartitioned_numel, world_size)


        tensor = GatheredTensor(fp32_flat_groups, flat_groups_offset, offset, partitioned_numel, shape)
        state_dict[name] = tensor
        offset += partitioned_numel

    offset *= world_size


    if offset != avail_numel:
        raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")

    print(f"Reconstructed Trainable fp32 state dict with {total_params} params {total_numel} elements")


def _get_fp32_state_dict_from_zero3_checkpoint(world_size, fp32_flat_groups, zero_model_states,
                                               exclude_frozen_parameters):
    state_dict = OrderedDict()


    buffers = zero_model_states[0].buffers
    state_dict.update(buffers)

    if not exclude_frozen_parameters:
        _zero3_merge_frozen_params(state_dict, world_size, zero_model_states)

    _zero3_merge_trainable_params(state_dict, world_size, fp32_flat_groups, zero_model_states)


    for pair in zero_model_states[0].shared_params:
        if pair[1] in state_dict:
            state_dict[pair[0]] = state_dict[pair[1]]

    return state_dict


def to_torch_tensor(state_dict, return_empty_tensor=False):
    """
    Convert state_dict of GatheredTensor to torch tensor
    """
    torch_state_dict = {}
    converted_tensors = {}
    for name, tensor in state_dict.items():
        tensor_id = id(tensor)
        if tensor_id in converted_tensors:  # shared tensors
            shared_tensor = torch_state_dict[converted_tensors[tensor_id]]
            torch_state_dict[name] = shared_tensor
        else:
            converted_tensors[tensor_id] = name
            if return_empty_tensor:
                torch_state_dict[name] = torch.empty(tensor.shape, dtype=tensor.dtype)
            else:
                torch_state_dict[name] = tensor.contiguous()
    return torch_state_dict


def get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir,
                                             tag=None,
                                             exclude_frozen_parameters=False,
                                             lazy_mode=False):
    """
    Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated state_dict that can be loaded with
    ``load_state_dict()`` and used for training without DeepSpeed or shared with others, for example
    via a model hub.

    Args:
        - ``checkpoint_dir``: path to the desired checkpoint folder
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in 'latest' file. e.g., ``global_step14``
        - ``exclude_frozen_parameters``: exclude frozen parameters
        - ``lazy_mode``: get state_dict in lazy mode. It returns a dict of pesduo tensor instead of torch tensor, which is more memory efficient.
          Convert the pesduo tensor to torch tensor by ``.contiguous()``

    Returns:
        - pytorch ``state_dict``

    A typical usage might be ::

        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir) # already on cpu
        model = model.cpu() # move to cpu
        model.load_state_dict(state_dict)


    In this example the ``model`` will no longer be usable in the deepspeed context of the same
    application. i.e. you will need to re-initialize the deepspeed engine, since
    ``model.load_state_dict(state_dict)`` will remove all the deepspeed magic from it.

    If you want it all done for you, use ``load_state_dict_from_zero_checkpoint`` instead.

    Note: the above usage may not work if your application doesn't have sufficient free CPU memory.
    You may need to use the offline approach using the ``zero_to_fp32.py`` script that is saved with
    the checkpoint. Or you can load state_dict in lazy mode ::

        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, lazy_mode=True) # not on cpu
        for name, lazy_tensor in state_dict.item():
            tensor = lazy_tensor.contiguous()  # to cpu
            print(name, tensor)

    """
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, 'latest')
        if os.path.isfile(latest_path):
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)

    if not os.path.isdir(ds_checkpoint_dir):
        raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")

    state_dict = _get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir, exclude_frozen_parameters)
    if lazy_mode:
        return state_dict
    else:
        return to_torch_tensor(state_dict)


def convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir,
                                               output_dir,
                                               max_shard_size="5GB",
                                               safe_serialization=False,
                                               tag=None,
                                               exclude_frozen_parameters=False):
    """
    Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated ``state_dict`` file that can be
    loaded with ``torch.load(file)`` + ``load_state_dict()`` and used for training without DeepSpeed.

    Args:
        - ``checkpoint_dir``: path to the desired checkpoint folder. (one that contains the tag-folder, like ``global_step14``)
        - ``output_dir``: directory to the pytorch fp32 state_dict output files
        - ``max_shard_size``: the maximum size for a checkpoint before being sharded, default value is 5GB
        - ``safe_serialization``:  whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in the file named ``latest`` in the checkpoint folder, e.g., ``global_step14``
        - ``exclude_frozen_parameters``: exclude frozen parameters
    """


    if safe_serialization:
        try:
            from safetensors.torch import save_file
        except ImportError:
            print('If you want to use `safe_serialization`, please `pip install safetensors`')
            raise
    if max_shard_size is not None:
        try:
            from huggingface_hub import split_torch_state_dict_into_shards
        except ImportError:
            print('If you want to use `max_shard_size`, please `pip install huggingface_hub`')
            raise


    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir,
                                                          tag,
                                                          exclude_frozen_parameters,
                                                          lazy_mode=True)


    weights_name = "model.safetensors" if safe_serialization else "pytorch_model.bin"
    if max_shard_size is not None:
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")

        empty_state_dict = to_torch_tensor(state_dict, return_empty_tensor=True)
        state_dict_split = split_torch_state_dict_into_shards(empty_state_dict,
                                                              filename_pattern=filename_pattern,
                                                              max_shard_size=max_shard_size)
    else:
        from collections import namedtuple
        StateDictSplit = namedtuple("StateDictSplit", ["is_sharded", "filename_to_tensors"])
        state_dict_split = StateDictSplit(is_sharded=False,
                                          filename_to_tensors={weights_name: list(state_dict.keys())})


    os.makedirs(output_dir, exist_ok=True)
    filename_to_tensors = state_dict_split.filename_to_tensors.items()
    for shard_file, tensors in tqdm(filename_to_tensors, desc="Saving checkpoint shards"):
        shard_state_dict = {tensor_name: state_dict[tensor_name] for tensor_name in tensors}
        shard_state_dict = to_torch_tensor(shard_state_dict)
        output_path = os.path.join(output_dir, shard_file)
        if safe_serialization:
            save_file(shard_state_dict, output_path, metadata={"format": "pt"})
        else:
            torch.save(shard_state_dict, output_path)

        for tensor_name in list(shard_state_dict.keys()):
            del state_dict[tensor_name]
            del shard_state_dict[tensor_name]
        del shard_state_dict
        gc.collect()


    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        save_index_file = "model.safetensors.index.json" if safe_serialization else "pytorch_model.bin.index.json"
        save_index_file = os.path.join(output_dir, save_index_file)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)


def load_state_dict_from_zero_checkpoint(model, checkpoint_dir, tag=None):
    """
    1. Put the provided model to cpu
    2. Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated ``state_dict``
    3. Load it into the provided model

    Args:
        - ``model``: the model object to update
        - ``checkpoint_dir``: path to the desired checkpoint folder. (one that contains the tag-folder, like ``global_step14``)
        - ``tag``: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in the file named ``latest`` in the checkpoint folder, e.g., ``global_step14``

    Returns:
        - ``model`: modified model

    Make sure you have plenty of CPU memory available before you call this function. If you don't
    have enough use the ``zero_to_fp32.py`` utility to do the conversion. You will find it
    conveniently placed for you in the checkpoint folder.

    A typical usage might be ::

        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)


    Note, that once this was run, the ``model`` will no longer be usable in the deepspeed context
    of the same application. i.e. you will need to re-initialize the deepspeed engine, since
    ``model.load_state_dict(state_dict)`` will remove all the deepspeed magic from it.

    """
    logger.info(f"Extracting fp32 weights")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)

    logger.info(f"Overwriting model with fp32 weights")
    model = model.cpu()
    model.load_state_dict(state_dict, strict=False)

    return model


def train():
    parser = argparse.ArgumentParser(description='DeepSpeed Training')
    parser.add_argument('--local_rank', type=int, default=-1, # local_rank
                    help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser) #  deepspeed  (deepspeed_config etc.)
    args = parser.parse_args() #
    with open(args.deepspeed_config, 'r', encoding='utf-8') as f:
        config = json.load(f) #  df_config.json

    if args.local_rank==0:
        logging.basicConfig(
            filename='./train.log',
            filemode='w',  #
            format='[%(asctime)s] [%(levelname)s] %(message)s', # message
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO
        )
        logger = logging.getLogger(__name__)

    deepspeed.utils.logger.setLevel(logging.WARNING)  # INFOdeepspeed(run.log)

    get_accelerator().set_device(args.local_rank) #
    device = torch.device(get_accelerator().device_name(), args.local_rank) #
    world_size = int(os.getenv("WORLD_SIZE", 1)) #

    if args.local_rank==0 and os.path.exists("./profile_logs"):
        shutil.rmtree("./profile_logs")

    save_path = config['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.basename(__file__)
    save_path = os.path.join(save_path, 'result_' + file_name.replace('.py', ''))

    if args.local_rank==0 and os.path.exists(save_path):
        shutil.rmtree(save_path)

    if args.local_rank==0:
        logger.info("========== arg1 ==========")
        logger.info(json.dumps(vars(args), indent=4, ensure_ascii=False))
        logger.info("========== arg2 ==========")
        logger.info(json.dumps(config, indent=4, ensure_ascii=False))
        logger.info("=============================")

    seed = 100 + args.local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = MyModel(n_filters=config['n_filters']).to(device=device)
    if args.local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params}")

    engine, _, _, _ = deepspeed.initialize(
        args=args, # parser(deepspeed_config)
        model=model,
        model_parameters=model.parameters(),
        dist_init_required=True # ()
    )
    custom_loss = MyLoss().to(device=device)

    train_dataset = MyDataset(config['train_data_path'])
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_micro_batch_size_per_gpu'],
        sampler=train_sampler,
        num_workers=max(4, os.cpu_count() // world_size),  # workers
        pin_memory=True,  #
        persistent_workers=True,  # workers
    )
    val_dataset = MyDataset(config['val_data_path'])
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train_micro_batch_size_per_gpu'],
        sampler=val_sampler,
        num_workers=max(4, os.cpu_count() // world_size),  # workers
        pin_memory=True,  #
        persistent_workers=True  # workers
    )

    if args.local_rank == 0:
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")



    val_best_loss = float('inf')
    no_improvement_count = 0
    es_patience = config.get('es_patience', 20)
    es_min_delta = config.get('es_min_delta', 1e-6)


    total_time_start = default_timer()

    for ep in range(config['epochs']):
        engine.train()
        train_sampler.set_epoch(ep)
        train_epoch_loss = 0.0
        ep_start_time = default_timer()



        if (ep >= config["profiler_ep_start"]) and (ep <= config["profiler_ep_end"]):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=1,     # 1
                    warmup=1,   # 1
                    active=3,   # 3
                    repeat=1    # 1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f'./profile_logs/rank{args.local_rank}_epoch{ep}'
                ),
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                for inputs, labels, targets in train_loader:
                    inputs = inputs.to(device=device, non_blocking=True)
                    labels = labels.to(device=device, non_blocking=True)
                    targets = targets.to(device=device, non_blocking=True)

                    with torch.cuda.amp.autocast():
                        outputs = engine(inputs, p=labels)
                        loss = custom_loss(outputs, targets)
                        if torch.isnan(loss).any():
                            logger.error("Error: Loss is NaN. Training terminated.")
                            raise ValueError("Loss is NaN")  #
                    engine.backward(loss)
                    engine.step()
                    train_epoch_loss += loss.item()
                    torch.cuda.synchronize() #
                    prof.step()
        else:
            for inputs, labels, targets in train_loader:
                inputs = inputs.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = engine(inputs, p=labels)
                    loss = custom_loss(outputs, targets)
                    if torch.isnan(loss).any():
                        logger.error("Loss is NaN. Training terminated.")
                        raise ValueError("Loss is NaN")  #
                engine.backward(loss)
                engine.step()
                train_epoch_loss += loss.item()

        epoch_time = default_timer() - ep_start_time


        train_loss_tensor = torch.tensor(train_epoch_loss, device=device)
        local_train_batches = torch.tensor(len(train_loader), device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_train_batches, op=dist.ReduceOp.SUM)

        train_avg_loss = train_loss_tensor.item() / local_train_batches.item()

        if args.local_rank == 0:
            logger.info(
                f"Epoch: {ep:>4d} | "
                f"Time: {epoch_time:>5.1f}s | "
                f"Train Loss: {train_avg_loss:>10.6e} | "
                f"LR: {engine.optimizer.param_groups[0]['lr']:>7.2e}"
            )


        if ep >= config.get('val_start_epoch', 0):
            engine.eval()
            val_loss_total = 0.0
            val_start_time = default_timer()
            with torch.no_grad():
                for inputs, labels, targets in val_loader:
                    inputs = inputs.to(device=device, non_blocking=True)
                    labels = labels.to(device=device, non_blocking=True)
                    targets = targets.to(device=device, non_blocking=True)

                    with torch.cuda.amp.autocast():
                        outputs = engine(inputs, p=labels)
                        loss = custom_loss(outputs, targets)
                        if torch.isnan(loss).any():
                            logger.error("Loss is NaN. Training terminated.")
                            raise ValueError("Loss is NaN")  #
                    val_loss_total += loss.item()
            val_time = default_timer() - val_start_time


            val_loss_tensor = torch.tensor(val_loss_total, device=device)
            local_val_batches = torch.tensor(len(val_loader), device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_batches, op=dist.ReduceOp.SUM)

            val_avg_loss = val_loss_tensor.item() / local_val_batches.item()

            if args.local_rank == 0:
                logger.info(
                    f"Epoch: {ep:>4d} | "
                    f"Time: {val_time:>5.1f}s | "
                    f"Val   Loss: {val_avg_loss:>10.6e} | "
                )


            if args.local_rank == 0:
                should_save = (val_avg_loss < val_best_loss - es_min_delta)
            else:
                should_save = False
            should_save_tensor = torch.tensor(int(should_save), device=device)
            dist.broadcast(should_save_tensor, src=0)


            if should_save_tensor.item() == 1:
                val_best_loss = val_avg_loss
                no_improvement_count = 0

                tag_best = "ds_best"
                engine.save_checkpoint(
                    save_dir=save_path,
                    tag=tag_best,
                    client_state={'epoch': ep, 'global_steps': engine.global_steps, 'config': config},
                    exclude_frozen_parameters=False,
                    save_latest=False
                )
                if engine.global_rank == 0:
                    convert_zero_checkpoint_to_fp32_state_dict(
                        checkpoint_dir=save_path,
                        output_dir=os.path.join(save_path,tag_best),
                        max_shard_size="5GB",
                        safe_serialization=False,
                        tag=tag_best,
                        exclude_frozen_parameters=False)
                    logger.info(f"[ best ] Checkpoint saved at epoch {ep}")
            else:
                no_improvement_count += 1


            no_improvement_tensor = torch.tensor(no_improvement_count, device=device)
            dist.all_reduce(no_improvement_tensor, op=dist.ReduceOp.MAX)
            no_improvement_count = no_improvement_tensor.item()


            if no_improvement_count >= es_patience:
                if engine.global_rank == 0:
                    logger.info(f"[stop]   Early stopping triggered at epoch {ep}")
                stop_training = torch.tensor(1, device=device)
            else:
                stop_training = torch.tensor(0, device=device)
            dist.all_reduce(stop_training, op=dist.ReduceOp.MAX)
            if stop_training.item() == 1:
                break


        if (ep + 1) % config['save_interval'] == 0:
            tag_latest = "ds_latest"
            engine.save_checkpoint( # rank
                save_dir=save_path,
                tag=tag_latest,
                client_state={
                    'epoch': ep,
                    'global_steps': engine.global_steps, #
                    'config': config  #
                },
                exclude_frozen_parameters=False,  # requires_grad=False
                save_latest=False #  "latest"
            )

            if engine.global_rank == 0:
                convert_zero_checkpoint_to_fp32_state_dict(
                    checkpoint_dir=save_path,
                    output_dir=os.path.join(save_path,tag_latest),
                    max_shard_size="5GB",
                    safe_serialization=False,
                    tag=tag_latest,
                    exclude_frozen_parameters=False)
                logger.info(f"[latest] Checkpoint saved at epoch {ep}")


    if args.local_rank == 0:
        total_hours = (default_timer() - total_time_start) / 3600
        logger.info(f"Training completed in {total_hours:.2f} hours.")


    tag_latest = "ds_latest"
    engine.save_checkpoint( # rank
        save_dir=save_path,
        tag=tag_latest,
        client_state={
            'epoch': ep,
            'global_steps': engine.global_steps, #
            'config': config  #
        },
        exclude_frozen_parameters=False,  # requires_grad=False
        save_latest=False #  "latest"
    )

    if engine.global_rank == 0:
        convert_zero_checkpoint_to_fp32_state_dict(
            checkpoint_dir=save_path,
            output_dir=os.path.join(save_path,tag_latest),
            max_shard_size="5GB",
            safe_serialization=False,
            tag=tag_latest,
            exclude_frozen_parameters=False)
        logger.info("[latest] Checkpoint saved at last")


    dist.destroy_process_group()

if __name__ == "__main__":
    train()
