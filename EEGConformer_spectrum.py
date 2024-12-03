"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""
# remember to change paths

import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('./TensorBoardX/')


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, in_channels=1, eeg_channels=8):
        super().__init__()

        self.shallownet = nn.Sequential(
            # 感覺可以 1
            # nn.Conv2d(4, 40, (1, 25), (1, 1)),
            # nn.Conv2d(40, 40, (1, 12), (1, 1)),      
            #      
            # 感覺可以 2
            # nn.Conv2d(4, 40, (1, 12), (1, 1)),
            # nn.Conv2d(40, 40, (1, 6), (1, 1)),
            # nn.Conv2d(40, 40, (1, 3), (1, 1)), 

            # nn.Conv3d(in_channels, 20, (1, 1, 4), (1, 1, 1)),
            # nn.BatchNorm3d(20), 
            # nn.Conv3d(20, 20, (1, 3, 1), (1, 1, 1)),
            # nn.BatchNorm3d(20), 
            # # nn.Conv2d(20, 40, (eeg_channels, 1), (1, 1)),
            # nn.Conv3d(20, 40, (eeg_channels, 1, 1), (1, 1, 1)),
            # nn.BatchNorm3d(40),                        
            # nn.ELU(),
            # nn.AvgPool3d((1, 1, 18), (1, 1, 1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT


            nn.Conv3d(in_channels, 40, (1, 3, 25), (1, 1, 1)),
            # nn.BatchNorm3d(20), 
            # nn.Conv3d(20, 20, (1, 3, 1), (1, 1, 1)),
            # nn.BatchNorm3d(20), 
            # nn.Conv2d(20, 40, (eeg_channels, 1), (1, 1)),
            nn.Conv3d(40, 40, (eeg_channels, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(40),                        
            nn.ELU(),
            nn.AvgPool3d((1, 1, 56), (1, 1, 4)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT

            # nn.ELU(),
            nn.Dropout(0.5)
        )

        self.projection = nn.Sequential(
            nn.Conv3d(40, emb_size, (1, 1, 1), stride=(1, 1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) s (w) -> b s (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _, _ = x.shape
        x = self.shallownet(x)
        # print("shallownet ", x.shape)
        x = self.projection(x)
        # print("projection ", x.shape)
        # exit()
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b s n (h d) -> b h s n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b s n (h d) -> b h s n d", h=self.num_heads)
        values = rearrange(self.values(x), "b s n (h d) -> b h s n d", h=self.num_heads)
        energy = torch.einsum('bhsqd, bhskd -> bhsqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhsal, bhslv -> bhsav ', att, values)
        out = rearrange(out, "b h s n d -> b s n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, in_size, n_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, 126),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(126, 16),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(16, n_classes)
        )
    def forward(self, x):        
        x = x.contiguous().view(x.size(0), -1)  
        # print(x.shape)
        out = self.fc(x)
        return x, out

class BinaryClassificationHead(nn.Sequential):
    def __init__(self, in_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, 126),
            nn.ELU(),
            nn.Linear(126, 16),
            nn.ELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):        
        x = x.contiguous().view(x.size(0), -1)  
        out = self.fc(x)
        return x, out


class MultiClassificationHead(nn.Sequential):
    def __init__(self, input_size = 4, n_classes = 4):
        super().__init__()

        self.fc = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 8),            
            nn.LeakyReLU(),
            nn.Linear(8, n_classes)
        )
        
    def forward(self, x):        
        out = self.fc(x)
        return out      

# 
# in_size 1640
# n_classes 4
# 
class Conformer(nn.Sequential):
    def __init__(self, in_channels=1, eeg_channels=8, emb_size=40, depth=6, in_size=1640,  n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, in_channels=in_channels, eeg_channels=eeg_channels),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(in_size, n_classes)
        )

class Conformer_Binary(nn.Sequential):
    def __init__(self, in_channels=1, eeg_channels=8, emb_size=40, depth=6, in_size=1640, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, in_channels=in_channels, eeg_channels=eeg_channels),
            TransformerEncoder(depth, emb_size),
            BinaryClassificationHead(in_size)
        )


if __name__ == '__main__':
    from torchsummary import summary

    model = Conformer(in_channels=1, eeg_channels=8, in_size=496, n_classes=2, depth=1).cuda() # in_size = 3600
    print(summary(model, (1, 8, 3, 200)))

    # for param in model.parameters():
    #     param.requires_grad = True

    # for name, param in model.named_parameters():
    #     # if name not in ['block2.1.weight', 'block2.1.bias', 'dense.weight', 'dense.bias', 'residual_block2.conv1.weight',
    #     #                 'residual_block2.bn1.weight', 'residual_block2.bn1.bias', 'residual_block2.conv2.weight', 'residual_block2.bn2.weight', 'residual_block2.bn2.bias']:
    #     if name in ['0.shallownet.0.weight', '0.shallownet.0.bias',
    #                 '0.shallownet.1.weight', '0.shallownet.1.bias',
    #                 '0.shallownet.2.weight', '0.shallownet.2.bias',
    #                 '0.shallownet.3.weight', '0.shallownet.3.bias',
    #                 '0.projection.0.weight', '0.projection.0.bias']:
    #         param.requires_grad = False

    # for name, param in model.named_parameters():
    #     print("name: ", name)
    #     print("requires_grad: ", param.requires_grad)

