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

# buffering interpolation kernels significantly speeds up computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
offset_summary = torch.tensor([[[0,0],[1,0]],[[0,1],[1,1]]]).unsqueeze(0).permute(0,3,2,1).to(device)
kernel_buffer_velocity = {}
kernel_buffer_velocity_superres = {}
kernel_buffer_pressure = {}
kernel_buffer_pressure_superres = {}
kernel_buffer_wave = {}
kernel_buffer_wave_superres = {}

def save_buffers():
	os.makedirs('Logger/buffers',exist_ok=True)
	path = 'Logger/buffers/buffers.dic'
	with open(path,"wb") as f:
		pickle.dump({"vel":kernel_buffer_velocity,"vel_superres":kernel_buffer_velocity_superres,"pres":kernel_buffer_pressure,"pres_superres":kernel_buffer_pressure_superres,"wave":kernel_buffer_wave,"wave_superres":kernel_buffer_wave_superres},f)

def load_buffers():
	global kernel_buffer_velocity,kernel_buffer_velocity_superres,kernel_buffer_pressure,kernel_buffer_pressure_superres,kernel_buffer_wave,kernel_buffer_wave_superres
	path = 'Logger/buffers/buffers.dic'
	with open(path,"rb") as f:
		buffers = pickle.load(f)
		kernel_buffer_velocity = buffers["vel"]
		kernel_buffer_velocity_superres = buffers["vel_superres"]
		kernel_buffer_pressure = buffers["pres"]
		kernel_buffer_pressure_superres = buffers["pres_superres"]
		kernel_buffer_wave = buffers["wave"]
		kernel_buffer_wave_superres = buffers["wave_superres"]

try:
	load_buffers()
	print("loaded buffers")
except:
	print("no buffers available")

def interpolate_states(hidden, offset, orders_v=[2,2], create_graph=True, retain_graph=True):
    """
    Interpolate hidden states in space/time and return fields needed for NS loss.

    Returns (shapes are (B, H, W, C)):
        ω, v, dω_dt, grad_ω, lap_ω, ψ
    Note: ψ (streamfunction) is appended as the 6th output to keep backward compatibility
    with existing code that unpacks the first five values.
    """
	
    # unpack dei 6 output (first channel is the scalar potential ψ)
    ψ, lap_ψ = interpolate_2d_velocity(
        hidden, offset, orders_v, create_graph, retain_graph
    )

    # Reorder to (B, H, W, C)
    ψ = ψ.permute(0,2,3,1)
    lap_ψ  = lap_ψ.permute(0, 2, 3, 1)

    return ψ[...,0], lap_ψ[...,0] 


def interpolate_2d_velocity(weights, offsets, orders=[2,2], create_graph=True, retain_graph=True):
    B, C, H, W = weights.shape
    offset_key = f"{float(offsets[0]):.6f} {float(offsets[1]):.6f}, orders:{orders}, res:{H}x{W}"

    if offset_key in kernel_buffer_velocity:
        kernels = kernel_buffer_velocity[offset_key]
    else:
        with torch.enable_grad():
            offs = (offsets.clone().unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2) - offset_summary)
            offs = offs.unsqueeze(2).unsqueeze(3).repeat(1,1,(orders[0]+1),(orders[1]+1),1,1)
            offs = offs.detach().requires_grad_(True)

            kernels = torch.zeros(1, 2, (orders[0]+1), (orders[1]+1), 2, 2,
                                   device=device, dtype=weights.dtype)

            # canale 0: ψ
            for l in range(orders[0]+1):
                for m in range(orders[1]+1):
                    kernels[0:1, 0:1, l, m, :, :] = p_multidim(
                        offs[:, :, l, m], [orders[0], orders[1]], [l, m]
                    )

            # canale 1: Δψ (offset-coords)
            lap_off = div(grad(kernels[0:1, 0:1, :, :, :, :], offs, create_graph, retain_graph),
                          offs, create_graph, retain_graph)

            kernels[0:1, 1:2] = lap_off /(H*H)
            kernels = kernels.reshape(1, 2, (orders[0]+1)*(orders[1]+1), 2, 2).detach()

            kernel_buffer_velocity[offset_key] = kernels
            save_buffers()
    
    # NIENTE crop: usa il campo completo e pad coerente con le BC
    #output = F.conv2d(F.pad(weights, pad=(1,1,1,1), mode='circular'), kernels[0])
    output = F.conv2d(F.pad(weights, pad=(1,1,1,1), mode='replicate', value=0.0), kernels[0])
    lap = weights[:, 1:2]
    return weights[:, 0:1], lap


