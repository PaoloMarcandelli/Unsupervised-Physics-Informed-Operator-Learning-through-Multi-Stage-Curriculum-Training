import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *

################################################################
#  1d fourier layer (aggiornata a torch.fft.rfft/irfft)
################################################################
class SpectralConv1d(nn.Module):
    """
    FNO 1D con pesi complessi e torch.fft (API moderne).
    Moltiplica solo i primi `modes1` coefficienti nello spettro.
    """
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # <= N//2 + 1

        scale = 1.0 / (in_channels * out_channels)
        # Parametro complesso: (Cin, Cout, modes1)
        # (userà cfloat o cdouble a seconda del dtype di x)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul1d(x_ft, w):
        """
        x_ft: (B, Cin, K) complex
        w   : (Cin, Cout, K) complex
        ->   : (B, Cout, K) complex
        """
        return torch.einsum("bik,iok->bok", x_ft, w)

    def forward(self, x):
        """
        x: (B, Cin, N) real
        -> (B, Cout, N) real
        """
        B, Cin, N = x.shape
        # FFT (norm="ortho" ≈ normalized=True delle vecchie API)
        x_ft = torch.fft.rfft(x, n=N, dim=-1, norm="ortho")  # (B, Cin, K) complex, K = N//2+1

        K = x_ft.size(-1)
        m = min(self.modes1, K)

        out_ft = torch.zeros(B, self.out_channels, K, dtype=x_ft.dtype, device=x.device)
        W = self.weights1.to(x_ft.dtype)  # cfloat/cdouble coerente con x_ft

        # Moltiplico solo le prime m frequenze
        out_ft[..., :m] = self._compl_mul1d(x_ft[..., :m], W[..., :m])

        # iFFT (stessa normalizzazione)
        x_out = torch.fft.irfft(out_ft, n=N, dim=-1, norm="ortho")  # (B, Cout, N) real
        return x_out


################################################################
#  1d SimpleBlock e Net1d (invariati)
################################################################
class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(SimpleBlock1d, self).__init__()

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm1d(self.width)
        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # x: (B, N, 2)
        
        x = self.fc0(x)          # (B, N, width)
    
        x = x.permute(0, 2, 1)   # (B, width, N)

        x1 = self.conv0(x); x2 = self.w0(x)
        x = self.bn0(x1 + x2); x = F.relu(x)

        x1 = self.conv1(x); x2 = self.w1(x)
        x = self.bn1(x1 + x2); x = F.relu(x)

        x1 = self.conv2(x); x2 = self.w2(x)
        x = self.bn2(x1 + x2); x = F.relu(x)

        x1 = self.conv3(x); x2 = self.w3(x)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 1)   # (B, N, width)
        x = self.fc1(x); x = F.relu(x)
        x = self.fc2(x)
        x = x.permute(0,2,1)          # (B, 3,N)
        return x

class Net1d(nn.Module):
    def __init__(self, modes, width):
        super(Net1d, self).__init__()
        self.conv1 = SimpleBlock1d(modes, width)

    def forward(self, x):
        # x: (B, N, 2)
        x = self.conv1(x)
        return x.squeeze(-1)     # (B, N)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c