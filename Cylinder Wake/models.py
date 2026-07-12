import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utilities3 import *
from unet_parts import *
from spline_models import interpolate_states
from setups import Dataset
from orthogonal import OrthoExpSpectralConv2dFast as QLayer

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
        self.fc0 = nn.Linear(12, self.width)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 9) #9 spline coeffs

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

class Quantum_SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(Quantum_SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(12, self.width)

        self.conv0 = QLayer(self.width, self.width, self.modes1, self.modes2, share_across_freq=False)
        self.conv1 = QLayer(self.width, self.width, self.modes1, self.modes2, share_across_freq=False)
        self.conv2 = QLayer(self.width, self.width, self.modes1, self.modes2, share_across_freq=False)
        self.conv3 = QLayer(self.width, self.width, self.modes1, self.modes2, share_across_freq=False)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 9) #9 spline coeffs

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
        # pad = (pad_left, pad_right, pad_top, pad_bottom)
        x = F.pad(x, (0, 1, 0, 1), mode='replicate')
        
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
    

class QNet2d(nn.Module):
    def __init__(self, modes, width):
        super(QNet2d, self).__init__()

        self.conv1 = Quantum_SimpleBlock2d(modes, modes, width)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


def toCuda(x):
    if isinstance(x, (tuple, list)):
        return [toCuda(xi) for xi in x]
    return x.cuda(non_blocking=True)


import numpy as np
import torch
import torch.nn as nn
from unet_parts import DoubleConv, Down, Up, OutConv

class fluid_model(nn.Module):
    def __init__(
        self,
        orders_v,
        hidden_size=64,
        interpolation_size=5,
        bilinear=True,
        input_size=12,
        residuals=False
    ):
        """
        :orders_v: order of spline for velocity potential (should be at least 2)
        :hidden_size: hidden size of neural net
        :interpolation_size: size of first interpolation layer for v_cond and v_mask
        """
        super(fluid_model, self).__init__()
        self.hidden_size = hidden_size
        self.bilinear = bilinear
        self.input_size = input_size

        self.orders_v = orders_v
        self.v_size = np.prod([i + 1 for i in orders_v])

        self.hidden_state_size = self.v_size
        self.residuals = residuals

        self.inc = DoubleConv(input_size, hidden_size)

        self.down1 = Down(hidden_size, 2 * hidden_size)
        self.down2 = Down(2 * hidden_size, 4 * hidden_size)
        self.down3 = Down(4 * hidden_size, 8 * hidden_size)
        factor = 2 if bilinear else 1
        self.down4 = Down(8 * hidden_size, 16 * hidden_size // factor)

        self.up1 = Up(16 * hidden_size, 8 * hidden_size // factor, bilinear)
        self.up2 = Up(8 * hidden_size, 4 * hidden_size // factor, bilinear)
        self.up3 = Up(4 * hidden_size, 2 * hidden_size // factor, bilinear)
        self.up4 = Up(2 * hidden_size, hidden_size, bilinear)
        
        self.outc = OutConv(hidden_size, self.hidden_state_size)

    def forward(self, x):
       
        x = x.permute(0,3,1,2)
        x1 = self.inc(x)

        x2 = self.down1(x1)
       

        x3 = self.down2(x2)
        

        x4 = self.down3(x3)
        

        x5 = self.down4(x4)
        

        x = self.up1(x5, x4)
        
        x = self.up2(x, x3)
        

        x = self.up3(x, x2)
        

        x = self.up4(x, x1)
        
        

        x = self.outc(x)

        x = torch.tanh(x)
        
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        x = x - mean
        
        return x[...,1:,1:]
