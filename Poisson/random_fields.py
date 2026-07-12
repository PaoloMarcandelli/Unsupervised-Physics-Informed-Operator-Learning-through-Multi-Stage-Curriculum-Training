import torch
import math

class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
        self.dim = dim
        self.device = device
        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))
        k_max = size // 2
        if dim == 1:
            k = torch.cat((torch.arange(0, k_max, device=device),
                           torch.arange(-k_max, 0, device=device)))
            eig = (size * math.sqrt(2.0) * sigma *
                   (4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            eig[0] = 0.0
            self.sqrt_eig = eig
        elif dim == 2:
            w = torch.cat((torch.arange(0, k_max, device=device),
                           torch.arange(-k_max, 0, device=device)), 0).repeat(size,1)
            kx = w.t(); ky = w
            eig = (size**2 * math.sqrt(2.0) * sigma *
                   (4*(math.pi**2)*(kx**2 + ky**2) + tau**2)**(-alpha/2.0))
            eig[0,0] = 0.0
            self.sqrt_eig = eig
        elif dim == 3:
            w = torch.cat((torch.arange(0, k_max, device=device),
                           torch.arange(-k_max, 0, device=device)), 0).repeat(size,size,1)
            kx = w.permute(1,2,0); ky = w; kz = w.permute(2,1,0)
            eig = (size**3 * math.sqrt(2.0) * sigma *
                   (4*(math.pi**2)*(kx**2 + ky**2 + kz**2) + tau**2)**(-alpha/2.0))
            eig[0,0,0] = 0.0
            self.sqrt_eig = eig
        self.size = tuple([size]*dim)

    def sample(self, N):
        # Draw independent real and imag noise
        noise_r = torch.randn(N, *self.size, device=self.device)
        noise_i = torch.randn(N, *self.size, device=self.device)
        # Complex coefficients with spectral filter
        coeff = (noise_r + 1j*noise_i) * self.sqrt_eig
        # Inverse FFT to physical space
        field = torch.fft.ifftn(coeff, dim=tuple(range(-self.dim, 0))).real
        return field
