import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utilities3 import *
from unet_parts import *
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
        # pad = (pad_left, pad_right, pad_top, pad_bottom)
        # x = F.pad(x, (0, 1, 0, 1), mode='replicate')
        x = x[...,1:,1:]
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
        residuals=False,
        zero_mean=True,
        use_tanh=True,
    ):
        """
        :orders_v: order of spline for velocity potential (should be at least 2)
        :hidden_size: hidden size of neural net
        :interpolation_size: size of first interpolation layer for v_cond and v_mask
        :zero_mean: subtract batch mean from spline coefficients (True for vorticity, False for density)
        :use_tanh: apply tanh to spline coefficients (True for vorticity, False for density)
        """
        super(fluid_model, self).__init__()
        self.hidden_size = hidden_size
        self.bilinear = bilinear
        self.input_size = input_size
        self.zero_mean = zero_mean
        self.use_tanh = use_tanh

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

        if self.use_tanh:
            x = 3.0 * torch.tanh(x / 3.0)  # soft clamp: linear near 0, saturates at ±9

        if self.zero_mean:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            x = x - mean

        return x[...,1:,1:]


class FourierFeatures(nn.Module):
    """
    Fourier feature mapping per il trunk (x,y[,scalari]) -> [sin,cos] a n_freq.
    """
    def __init__(self, in_dim: int, n_freq: int = 8, scale: float = 10.0, trainable: bool = False):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * scale
        self.B = nn.Parameter(B, requires_grad=trainable)
        self.out_dim = 2 * n_freq

    def forward(self, x):  # x: (..., in_dim)
        proj = 2.0 * math.pi * (x @ self.B)  # (..., n_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class BranchCNN2D(nn.Module):
    """
    Encoder CNN 2D per lo stack di vorticità (senza x,y).
    Input : (B,S,S,T_in)
    Output: (B, latent)
    """
    def __init__(self, in_ch: int, latent: int = 128, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  width,   kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(width,  width,   kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width,  2*width, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2*width,2*width, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> (B, 2*width, 1, 1)
        )
        self.proj = nn.Linear(2*width, latent)

    def forward(self, x):  # (B,S,S,T_in)
        x = x.permute(0, 3, 1, 2).contiguous()   # -> (B,T_in,S,S)
        h = self.net(x).flatten(1)               # (B, 2*width)
        b = self.proj(h)                         # (B, latent)
        return b


class TrunkMLP(nn.Module):
    """
    MLP su coordinate (x,y) + opzionali (dt, nu, extra scalars), per-pixel.
    Input : (B,S,S,D_in)
    Output: (B,S,S,latent)
    """
    def __init__(self, in_dim: int, latent: int = 128, hidden: int = 128, depth: int = 4,
                 use_fourier: bool = True, n_freq: int = 8, ff_scale: float = 10.0):
        super().__init__()
        self.use_fourier = use_fourier
        if self.use_fourier:
            self.ff = FourierFeatures(in_dim, n_freq=n_freq, scale=ff_scale, trainable=False)
            first = self.ff.out_dim
        else:
            self.ff = None
            first = in_dim

        layers = []
        dims = [first] + [hidden]*(depth-1) + [latent]
        for i in range(depth):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < depth - 1:
                layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords):  # (B,S,S,D_in)
        B, S, _, Din = coords.shape
        x = coords.reshape(B*S*S, Din)
        if self.use_fourier:
            x = self.ff(x)
        t = self.mlp(x)                   # (B*S*S, latent)
        return t.view(B, S, S, -1)        # (B,S,S,latent)


class DeepONet2D_NS_Vorticity(nn.Module):
    """
    DeepONet per NS (formulazione vorticità) one-step:
      Input  X: (B,S,S, 2+T_in)  -> [x,y | ω_{t−T_in+1: t}]
      Output Y: (B,S,S, 1)       -> ω̂_{t+1}

    Opzionali: passare dt, nu e un vettore di extra scalars al trunk (broadcast) per conditioning.
    """
    def __init__(self,
                 S: int,
                 T_in: int,
                 latent: int = 128,
                 branch_width: int = 64,
                 trunk_hidden: int = 128,
                 trunk_depth: int = 4,
                 trunk_use_fourier: bool = True,
                 trunk_nfreq: int = 8,
                 trunk_ff_scale: float = 10.0,
                 expect_dt: bool = False,
                 expect_nu: bool = False,
                 extra_dim: int = 0):
        super().__init__()
        self.S = S
        self.T_in = T_in
        self.latent = latent
        self.expect_dt = expect_dt
        self.expect_nu = expect_nu
        self.extra_dim = extra_dim

        # Branch: SOLO i T_in canali di ω (non include x,y)
        self.branch = BranchCNN2D(in_ch=T_in, latent=latent, width=branch_width)

        # Trunk: (x,y) + [dt] + [nu] + [extra scalars]
        trunk_in = 2 + (1 if expect_dt else 0) + (1 if expect_nu else 0) + extra_dim
        self.trunk = TrunkMLP(in_dim=trunk_in, latent=latent,
                              hidden=trunk_hidden, depth=trunk_depth,
                              use_fourier=trunk_use_fourier,
                              n_freq=trunk_nfreq, ff_scale=trunk_ff_scale)

        # Bias globale del dot-product
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor,
                dt: torch.Tensor = None,
                nu: torch.Tensor = None,
                extra: torch.Tensor = None):
        """
        X    : (B,S,S, 2+T_in) = [x,y | ω-stack]
        dt   : (B,) o (B,1)   opzionale se expect_dt=True
        nu   : (B,) o (B,1)   opzionale se expect_nu=True
        extra: (B,K)          opzionale con K=extra_dim
        """
        B, S, _, C = X.shape
        assert C == 2 + self.T_in, f"DeepONet: expected channels=2+T_in ({2+self.T_in}), got {C}"
        assert S == self.S, f"S mismatch: model S={self.S}, input S={S}"

        # Split ingresso
        xy   = X[..., :2]    # (B,S,S,2)
        wstk = X[..., 2:]    # (B,S,S,T_in)

        # Branch -> (B, latent)
        b = self.branch(wstk)

        # Trunk input per-pixel: concatena (x,y) + scalari broadcast
        pieces = [xy]
        if self.expect_dt:
            assert dt is not None, "DeepONet set expect_dt=True but dt is None"
            dtb = dt.view(B, 1, 1, 1).expand(B, S, S, 1)
            pieces.append(dtb)
        if self.expect_nu:
            assert nu is not None, "DeepONet set expect_nu=True but nu is None"
            nub = nu.view(B, 1, 1, 1).expand(B, S, S, 1)
            pieces.append(nub)
        if self.extra_dim > 0:
            assert (extra is not None) and (extra.shape[-1] == self.extra_dim), \
                f"extra must be (B,{self.extra_dim})"
            extb = extra.view(B, 1, 1, self.extra_dim).expand(B, S, S, self.extra_dim)
            pieces.append(extb)

        coords = torch.cat(pieces, dim=-1)   # (B,S,S,D_in)

        # Trunk -> (B,S,S,latent)
        t = self.trunk(coords)

        # DeepONet dot-product per-pixel: (B,1,1,p) · (B,S,S,p) -> (B,S,S,1)
        y = (t * b.view(B, 1, 1, -1)).sum(dim=-1, keepdim=True) + self.bias
        return y


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResNetBasicBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act(x + residual)
        return x


class ResNet18_NS(nn.Module):
    """
    ResNet-18 style baseline for one-step vorticity prediction.
    Input : (B,S,S,2+T_in) -> [x, y, ω-stack]
    Output: (B,S,S,1)      -> ω̂_{t+1}
    """
    def __init__(self, T_in: int, width: int = 64, residual: bool = True):
        super().__init__()
        in_ch = 2 + T_in
        self.T_in = T_in
        self.residual = residual

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=7, padding=3, bias=False),
            _group_norm(width),
            nn.GELU()
        )
        self.layers = nn.Sequential(*[ResNetBasicBlock2D(width) for _ in range(8)])
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            _group_norm(width),
            nn.GELU(),
            nn.Conv2d(width, 1, kernel_size=1)
        )

    def forward(self, x):
        x_in = x.permute(0, 3, 1, 2).contiguous()
        h = self.stem(x_in)
        h = self.layers(h)
        y = self.head(h).permute(0, 2, 3, 1).contiguous()
        if self.residual:
            y = y + x[..., -1:].contiguous()
        return y


class TFTemporalBlock(nn.Module):
    def __init__(self, t_in: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, width // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            _group_norm(width // 2),
            nn.GELU(),
            nn.Conv3d(width // 2, width, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            _group_norm(width),
            nn.GELU(),
        )
        self.pool_t = nn.Conv3d(width, width, kernel_size=(t_in, 1, 1), bias=False)
        self.norm = _group_norm(width)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = self.pool_t(x).squeeze(2)
        return self.act(self.norm(x))


class TFSpatialBlock(nn.Module):
    def __init__(self, in_ch: int, width: int):
        super().__init__()
        branch_width = width // 2
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_width, kernel_size=3, padding=1, bias=False),
            _group_norm(branch_width), nn.GELU())
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_width, kernel_size=5, padding=2, bias=False),
            _group_norm(branch_width), nn.GELU())
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_width, kernel_size=3, padding=2, dilation=2, bias=False),
            _group_norm(branch_width), nn.GELU())
        self.merge = nn.Sequential(
            nn.Conv2d(3 * branch_width, width, kernel_size=1, bias=False),
            _group_norm(width), nn.GELU())

    def forward(self, x):
        return self.merge(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class TFFusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _group_norm(channels), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _group_norm(channels))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TFNet_NS(nn.Module):
    """
    TF-Net-like baseline for turbulent flow prediction.
    Input : (B,S,S,2+T_in) -> [x, y, ω-stack]
    Output: (B,S,S,1)      -> ω̂_{t+1}
    """
    def __init__(self, T_in: int, width: int = 64, fusion_depth: int = 4, residual: bool = True):
        super().__init__()
        self.T_in = T_in
        self.residual = residual

        self.temporal = TFTemporalBlock(t_in=T_in, width=width)
        self.spatial = TFSpatialBlock(in_ch=2 + T_in, width=width)
        self.fusion_in = nn.Sequential(
            nn.Conv2d(2 * width, width, kernel_size=1, bias=False),
            _group_norm(width), nn.GELU())
        self.fusion = nn.Sequential(*[TFFusionBlock(width) for _ in range(fusion_depth)])
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            _group_norm(width), nn.GELU(),
            nn.Conv2d(width, 1, kernel_size=1))

    def forward(self, x):
        x_all = x.permute(0, 3, 1, 2).contiguous()
        omega_hist = x[..., 2:].permute(0, 3, 1, 2).contiguous()
        h_t = self.temporal(omega_hist)
        h_s = self.spatial(x_all)
        h = self.fusion_in(torch.cat([h_t, h_s], dim=1))
        h = self.fusion(h)
        y = self.head(h).permute(0, 2, 3, 1).contiguous()
        if self.residual:
            y = y + x[..., -1:].contiguous()
        return y


# ══════════════════════════════════════════════════════════════════════════════
# PINO  —  FNO backbone outputting 1 channel directly (no spline decoder)
# ══════════════════════════════════════════════════════════════════════════════

class _PINOSpectralConv2d(nn.Module):
    """Spectral convolution layer (rfft2, real-imag split) used by PINONet2d."""
    def __init__(self, in_ch, out_ch, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        self.in_channels  = in_ch
        self.out_channels = out_ch
        s = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(s * torch.rand(in_ch, out_ch, m1, m2, 2))
        self.w2 = nn.Parameter(s * torch.rand(in_ch, out_ch, m1, m2, 2))

    @staticmethod
    def _cmul(a, b):
        op = partial(torch.einsum, "bctq,dctq->bdtq")
        real = op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1])
        imag = op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):
        B, _, nx, ny = x.shape
        xf  = torch.fft.rfft2(x, norm='ortho')
        xf  = torch.stack([xf.real, xf.imag], dim=-1)
        out = torch.zeros(B, self.out_channels, nx, ny // 2 + 1, 2, device=x.device)
        out[:, :, :self.m1,  :self.m2] = self._cmul(xf[:, :, :self.m1,  :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = self._cmul(xf[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(torch.complex(out[..., 0], out[..., 1]),
                                s=(nx, ny), norm='ortho')


class PINONet2d(nn.Module):
    """FNO backbone that predicts the scalar field directly (1 output channel)."""
    def __init__(self, modes, width):
        super().__init__()
        self.width = width
        self.fc0   = nn.Linear(12, width)
        self.conv0 = _PINOSpectralConv2d(width, width, modes, modes)
        self.conv1 = _PINOSpectralConv2d(width, width, modes, modes)
        self.conv2 = _PINOSpectralConv2d(width, width, modes, modes)
        self.conv3 = _PINOSpectralConv2d(width, width, modes, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.bn0 = nn.BatchNorm2d(width)
        self.bn1 = nn.BatchNorm2d(width)
        self.bn2 = nn.BatchNorm2d(width)
        self.bn3 = nn.BatchNorm2d(width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        B, sx, sy, _ = x.shape
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = F.relu(self.bn0(self.conv0(x) + self.w0(x.view(B, self.width, -1)).view(B, self.width, sx, sy)))
        x = F.relu(self.bn1(self.conv1(x) + self.w1(x.view(B, self.width, -1)).view(B, self.width, sx, sy)))
        x = F.relu(self.bn2(self.conv2(x) + self.w2(x.view(B, self.width, -1)).view(B, self.width, sx, sy)))
        x =        self.bn3(self.conv3(x) + self.w3(x.view(B, self.width, -1)).view(B, self.width, sx, sy))
        x = x.permute(0, 2, 3, 1)
        return self.fc2(F.relu(self.fc1(x)))   # (B, S, S, 1)

    def count_params(self):
        return sum(reduce(operator.mul, p.size()) for p in self.parameters())