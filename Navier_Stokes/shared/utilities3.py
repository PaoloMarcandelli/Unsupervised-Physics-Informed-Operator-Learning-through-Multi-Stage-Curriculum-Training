import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
import torch.nn as nn
from scipy.ndimage import gaussian_filter


#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        # prova a leggere l’header per vedere se è HDF5 (v7.3)
        with open(self.file_path, 'rb') as f:
            header = f.read(8)
        if header.startswith(b'\x89HDF\r\n\x1a\n'):
            # HDF5-based MAT (v7.3)
            self.data = h5py.File(self.file_path, 'r')
            self.old_mat = False
        else:
            # altrimenti MAT-file v5
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True


    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/(y_norms)

    def __call__(self, x, y):
        return self.rel(x, y)
    
    # loss function with safe rel/abs Lp loss
class LpLossSafe(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, eps=1e-5, tau_switch=None):
        """
        eps: epsilon per evitare divisioni per zero nella loss relativa
        tau_switch: se non None, per ||y|| < tau_switch usa loss assoluta al posto della relativa
        """
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps = eps
        self.tau_switch = tau_switch

    def abs(self, x, y):
        B = x.size(0)
        # mesh uniforme; usa float del tensore per device/dtype corretti
        h = (x.new_tensor(1.0) / (x.size(1) - 1.0))
        norms = (h**(self.d/self.p)) * torch.norm(x.reshape(B, -1) - y.reshape(B, -1), self.p, dim=1)
        if not self.reduction:
            return norms
        return norms.mean() if self.size_average else norms.sum()

    def rel(self, x, y):
        B = x.size(0)
        diff = torch.norm(x.reshape(B, -1) - y.reshape(B, -1), self.p, dim=1)
        ynrm = torch.norm(y.reshape(B, -1), self.p, dim=1).clamp_min(self.eps)

        if self.tau_switch is not None:
            # per campioni con ||y|| molto piccola usa loss assoluta (evita rapporti instabili)
            use_abs = (ynrm < self.tau_switch)
            # costruisci loss per-campione
            abs_loss = torch.norm(x.reshape(B, -1) - y.reshape(B, -1), self.p, dim=1)
            rel_loss = diff / ynrm
            per_sample = torch.where(use_abs, abs_loss, rel_loss)
        else:
            per_sample = diff / ynrm

        if not self.reduction:
            return per_sample
        return per_sample.mean() if self.size_average else per_sample.sum()

    def __call__(self, x, y, mode="rel"):
        return self.rel(x, y) if mode == "rel" else self.abs(x, y)


# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x
