import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utilities3 import *
from spline_models import interpolate_states
from setups import Dataset

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import matplotlib.pyplot as plt
import os
import pandas as pd

from get_params import get_params

################################################################
# Fourier layer updated to torch.fft
################################################################

def compl_mul2d_rfft(a, b):
    # a: (batch, in_ch, m1, m2, 2)
    # b: (in_ch, out_ch, m1, m2, 2)
    op = partial(torch.einsum, "bctq,dctq->bdtq")
    real = op(a[...,0], b[...,0]) - op(a[...,1], b[...,1])
    imag = op(a[...,1], b[...,0]) + op(a[...,0], b[...,1])
    return torch.stack([real, imag], dim=-1)

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1/(in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))

    def forward(self, x):
        batchsize, _, nx, ny = *x.shape, 
        # Compute Fourier coefficients (real input -> rfft2)
        x_ft_complex = torch.fft.rfft2(x, norm='ortho')
        # Convert to real-imag tensor
        x_ft = torch.stack([x_ft_complex.real, x_ft_complex.imag], dim=-1)
        # Allocate output in Fourier space
        out_ft = torch.zeros(batchsize, self.out_channels, nx, ny//2+1, 2, device=x.device)
        # Multiply modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d_rfft(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d_rfft(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        # Convert back to complex
        out_ft_complex = torch.complex(out_ft[...,0], out_ft[...,1])
        # Return to physical space
        x = torch.fft.irfft2(out_ft_complex, s=(nx, ny), norm='ortho')
        return x


class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.bn0 = nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)
        self.bn1 = nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)
        self.bn2 = nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)
        self.bn3 = nn.InstanceNorm2d(self.width, affine=True, track_running_stats=False)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 25) #9 spline coeffs

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # ora sposto i 9 canali in testa per fare il pad spaziale
        x = x.permute(0, 3, 1, 2)        # (B, 9, H, W)
        
        return x


class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes, width)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
    


