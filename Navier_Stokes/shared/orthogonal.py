# orthogonal_exp.py
import torch
import torch.nn as nn

def upper_index(d, device=None):
    # Indici della metà superiore (esclusa diagonale)
    iu, ju = torch.triu_indices(d, d, offset=1, device=device)
    return iu, ju  # (M,), (M,) con M=d(d-1)/2

class OrthoExpSpectralConv2dFast(nn.Module):
    """
    Q = exp(A), A skew-symmetric costruito da M = d(d-1)/2 parametri.
    Applica Q direttamente ai canali della rFFT (niente W).
    - in_channels == out_channels == d
    - share_across_freq: un set di parametri per tutte le frequenze (più veloce)
    """
    def __init__(self, in_channels, out_channels, modes1, modes2,
                 share_across_freq=False, theta_scale=1e-3):
        super().__init__()
        assert in_channels == out_channels, "richiede mappa quadrata"
        self.d = in_channels
        self.m1, self.m2 = modes1, modes2
        self.share = share_across_freq

        M = self.d * (self.d - 1) // 2
        shape = (M,) if share_across_freq else (self.m1, self.m2, M)
        self.thetas_pos = nn.Parameter(theta_scale * torch.randn(*shape))
        self.thetas_neg = nn.Parameter(theta_scale * torch.randn(*shape))

        iu, ju = upper_index(self.d)
        self.register_buffer("iu", iu, persistent=False)
        self.register_buffer("ju", ju, persistent=False)

    def _build_Q(self, thetas: torch.Tensor, device, dtype):
        """
        Costruisce A skew-sim. e Q=exp(A).
        thetas: (M,) oppure (m1,m2,M). Restituisce Q: (m1,m2,d,d).
        """
        if thetas.dim() == 1:
            # condivisi: espandi senza copia sulla griglia di frequenze
            thetas = thetas.view(1, 1, -1).expand(self.m1, self.m2, -1)

        m1, m2, M = thetas.shape
        assert M == self.iu.numel()  # M = d(d-1)/2

        # Alloca A batchato sulle frequenze
        A = torch.zeros(m1, m2, self.d, self.d, device=device, dtype=dtype)
        # Riempie metà superiore/inferiore
        A[:, :, self.iu, self.ju] = thetas
        A[:, :, self.ju, self.iu] = -thetas

        # Exp di matrice (batchata): (m1,m2,d,d)
        Q = torch.linalg.matrix_exp(A)
        return Q

    def _apply_Q(self, xR, xI, thetas):
        # xR/xI: (B, d, m1, m2)
        Q = self._build_Q(thetas, xR.device, xR.dtype)  # (m1,m2,d,d)
        # out_d = sum_c in_c * Q[d,c]
        outR = torch.einsum('bctq,tqdc->bdtq', xR, Q)
        outI = torch.einsum('bctq,tqdc->bdtq', xI, Q)
        return outR, outI

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, d, nx, ny = x.shape
        assert d == self.d
        X = torch.fft.rfft2(x, norm='ortho')
        xR, xI = X.real, X.imag  # (B,d,nx,ny//2+1)

        # Seleziona blocchi di frequenze come nel tuo FNO
        xR_pos = xR[:, :, :self.m1, :self.m2].contiguous()
        xI_pos = xI[:, :, :self.m1, :self.m2].contiguous()
        xR_neg = xR[:, :, -self.m1:, :self.m2].contiguous()
        xI_neg = xI[:, :, -self.m1:, :self.m2].contiguous()

        yR_pos, yI_pos = self._apply_Q(xR_pos, xI_pos, self.thetas_pos)
        yR_neg, yI_neg = self._apply_Q(xR_neg, xI_neg, self.thetas_neg)

        # Rimetti al posto giusto
        xR[:, :, :self.m1, :self.m2]  = yR_pos
        xI[:, :, :self.m1, :self.m2]  = yI_pos
        xR[:, :, -self.m1:, :self.m2] = yR_neg
        xI[:, :, -self.m1:, :self.m2] = yI_neg

        return torch.fft.irfft2(torch.complex(xR, xI), s=(nx, ny), norm='ortho')
