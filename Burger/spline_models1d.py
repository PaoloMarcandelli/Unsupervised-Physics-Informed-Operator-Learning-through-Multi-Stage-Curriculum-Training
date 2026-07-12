import torch
from torch import nn
import torch.nn.functional as F
from operators import rot,grad,div
import numpy as np
import os,pickle


def sign(x):
	s = torch.sign(x)
	s[s==0]=1
	return s

def heaviside(x):
	return (torch.sign(x)+1)/2

# 1st order splines
def p1_1(offsets):
	offsets = offsets*sign(offsets)
	return (1-offsets)

p1 = [p1_1] # list of 1st order basis splines

# 2nd order splines
def p2_1(offsets):
	offsets = offsets*sign(offsets)
	return (1-offsets)**2*(1+2*offsets)

def p2_2(offsets):
	abs_offsets = offsets*sign(offsets)
	return sign(offsets)*(1-abs_offsets)**2*(abs_offsets)

# derivatives
def dp2_1(offsets):#first derivative (needs to be devided by dt)
	abs_offsets = offsets*sign(offsets)
	return sign(offsets)*(6*abs_offsets**2-6*abs_offsets)

def dp2_2(offsets):
	abs_offsets = offsets*sign(offsets)
	return 3*abs_offsets**2 - 4*abs_offsets + 1

def d2p2_1(offsets):#2nd derivative (needs to be devided by dt**2)
	abs_offsets = offsets*sign(offsets)
	return 12*abs_offsets - 6

def d2p2_2(offsets):
	abs_offsets = offsets*sign(offsets)
	return sign(offsets)*(6*abs_offsets-4)

p2 = [p2_1,p2_2] # list of 2nd order basis splines

# 3rd order splines
def p3_1(offsets):
	offsets = offsets*sign(offsets)
	return (1-offsets)**3*(1+3*offsets+6*offsets**2)

def p3_2(offsets):
	abs_offsets = offsets*sign(offsets)
	return sign(offsets)*(1-abs_offsets)**3*(abs_offsets+3*abs_offsets**2)*2

def p3_3(offsets):
	offsets = offsets*sign(offsets)
	return (1-offsets)**3*(0.5*offsets**2)*16

p3 = [p3_1,p3_2,p3_3] # list of 3rd order basis splines

# 4th order splines
def p4_1(offsets):
	return (offsets-1)**4*(1+4*offsets+10*offsets**2+20*offsets**3)*heaviside(offsets)+(-offsets-1)**4*(1-4*offsets+10*offsets**2-20*offsets**3)*heaviside(-offsets)

def p4_2(offsets):
	return ((offsets-1)**4*(1*offsets+4*offsets**2+10*offsets**3)*heaviside(offsets)+(-offsets-1)**4*(1*offsets-4*offsets**2+10*offsets**3)*heaviside(-offsets))*4

def p4_3(offsets):
	return ((offsets-1)**4*(0.5*offsets**2+2*offsets**3)*heaviside(offsets)+(-offsets-1)**4*(0.5*offsets**2-2*offsets**3)*heaviside(-offsets))*32

def p4_4(offsets):
	return ((offsets-1)**4*(1.0/6.0*offsets**3)*heaviside(offsets)+(-offsets-1)**4*(1.0/6.0*offsets**3)*heaviside(-offsets))*512

p4 = [p4_1,p4_2,p4_3,p4_4]

# 5th order splines
def p5_1(offsets):
	return ((offsets-1)**5*(-1-5*offsets-15*offsets**2-35*offsets**3-70*offsets**4)*heaviside(offsets)+(-offsets-1)**5*(-1+5*offsets-15*offsets**2+35*offsets**3-70*offsets**4)*heaviside(-offsets))

def p5_2(offsets):
	return ((offsets-1)**5*(-1*offsets-5*offsets**2-15*offsets**3-35*offsets**4)*heaviside(offsets)+(-offsets-1)**5*(-1*offsets+5*offsets**2-15*offsets**3+35*offsets**4)*heaviside(-offsets))*4

def p5_3(offsets):
	return ((offsets-1)**5*(-0.5*offsets**2-2.5*offsets**3-7.5*offsets**4)*heaviside(offsets)+(-offsets-1)**5*(-0.5*offsets**2+2.5*offsets**3-7.5*offsets**4)*heaviside(-offsets))*32

def p5_4(offsets):
	return ((offsets-1)**5*(-0.5/3.0*offsets**3-2.5/3.0*offsets**4)*heaviside(offsets)+(-offsets-1)**5*(-0.5/3.0*offsets**3+2.5/3.0*offsets**4)*heaviside(-offsets))*512

def p5_5(offsets):
	return ((offsets-1)**5*(-2.5/6.0*offsets**4)*heaviside(offsets)+(-offsets-1)**5*(-2.5/6.0*offsets**4)*heaviside(-offsets))*1024

p5 = [p5_1,p5_2,p5_3,p5_4,p5_5]

pi = [p1,p2,p3,p4,p5] # list of lists of basis splines for different orders

def p_multidim(offsets,orders,indices):
	"""
	multidimensional basis spline of specified orders and indices
	:offsets: offsets of size: bs x n_dims x ...
	:orders: orders of spline for each dimension (note: counting starts at 0 => 0 ~ 1st order, 1 ~ 2nd order, 2 ~ 3rd order)
	:indices: indices of spline for each dimension (note: counting starts at 0)
	"""
	return torch.prod(torch.cat([pi[orders[i]][indices[i]](offsets[:,i:(i+1)]).unsqueeze(0) for i in range(len(orders))]),dim=0)

import os, pickle, torch
import torch.nn.functional as F
from operators import grad           # la tua grad(x,y,...)
# p_multidim deve essere importata/definita in questo file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) stencil 1D: nodi {0,1}  -> offsets locali [δ-0, δ-1]
offset_summary = torch.tensor([[[0.0, 1.0]]], dtype=torch.get_default_dtype(), device=device)          # (1,1,2)

kernel_buffer_velocity = {}

def save_buffers():
    os.makedirs('Logger/buffers', exist_ok=True)
    with open('Logger/buffers/buffers_1d.dic',"wb") as f:
        pickle.dump({"vel": kernel_buffer_velocity}, f)

def load_buffers():
    global kernel_buffer_velocity
    with open('Logger/buffers/buffers_1d.dic',"rb") as f:
        kernel_buffer_velocity = pickle.load(f)["vel"]

try:
    load_buffers(); print("loaded 1D buffers")
except Exception:
    print("no 1D buffers available")

def _scalar_offset(offset):
    return float(offset.reshape(-1)[0].item()) if torch.is_tensor(offset) else float(offset)

def _build_kernels_1d(offset, order, create_graph, retain_graph):
    off = _scalar_offset(offset)
    key = f"off={off:.6f}|ord={order}|k2"
    if key in kernel_buffer_velocity:
        return kernel_buffer_velocity[key]

    L = len(pi[order])
    dtype = torch.get_default_dtype()

    base = (torch.tensor(off, device=device, dtype=dtype).view(1,1,1) - offset_summary)   # (1,1,2)
    offs = base.unsqueeze(2).repeat(1,1,L,1).detach().clone().requires_grad_(True)

    kernels = torch.zeros(1, 3, L, 2, device=device, dtype=dtype)

    # u: basi φ_l(δ)
    for l in range(L):
        kernels[0:1, 0:1, l, :] = p_multidim(offs[:, :, l], [order], [l])  # (1,1,2)

    # u_x, u_xx: derivate rispetto a offs
    kernels[0:1, 1:2] = grad(kernels[0:1, 0:1, ...], offs, create_graph=True,  retain_graph=True)
    kernels[0:1, 2:3] = grad(kernels[0:1, 1:2, ...], offs, create_graph=create_graph, retain_graph=retain_graph)

    kernels = kernels.detach()
    kernel_buffer_velocity[key] = kernels
    save_buffers()
    return kernels



def interpolate_1d_velocity(weights, offset, orders=[2], create_graph=True, retain_graph=True):
    """
    weights: (B, L, S) con L = order+1
    offset : scalare (0 se valuti sui nodi)
    ritorna: u, u_x, u_xx ciascuno (B,1,S)
    """
    assert len(orders) == 1
    order = int(orders[0])

    K = _build_kernels_1d(offset, order, create_graph, retain_graph)   # (1,3,L,2)

    # bordo come nel 2D: togli 1 e pad circolare di 1 per kernel_size=2
    w = F.pad(weights, pad=(0, 1), mode='circular')         # (B,L,S)

    out = F.conv1d(w, K[0], padding=0)                                 # (B,3,S)
    return out[:, 0:1, :], out[:, 1:2, :], out[:, 2:3, :]

def interpolate_states(hidden, offset, orders_v=[2], create_graph=True, retain_graph=True):
    u, u_x, u_xx = interpolate_1d_velocity(hidden, offset, orders_v, create_graph, retain_graph)
    return u[:,0,:], u_x[:,0,:], u_xx[:,0,:]


