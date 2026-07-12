import torch
import math

def grad(x,y,create_graph=False,retain_graph=False):
	"""
	compute gradients of x = f(y)
	:x: function outputs
	:y: parameters (expected shape: [batchsize x 3] - 3 for x,y,t)
	:return: dx/dy
	"""
	return torch.autograd.grad(torch.sum(x),y,create_graph=create_graph,retain_graph=retain_graph)[0]

def rot(x,y,create_graph=False,retain_graph=False):
	"""
	compute curl of x = f(y)
	:x: function outputs
	:y: parameters (expected shape: [batchsize x 3] - 3 for x,y,t)
	"""
	result = grad(x,y,create_graph,retain_graph)[:,[1,0]]
	result[:,1] = -result[:,1]
	return result


def laplace(x, y, create_graph = False,retain_graph=False):
	"""
	compute laplacian of x = f(y)
	:x: function outputs
	:y: parameters (expected shape: [batchsize x 3] - 3 for x,y,t)
	"""
	return div(grad(x,y,create_graph=True,retain_graph=True), y,create_graph=create_graph,retain_graph=retain_graph)

def div(x, y,create_graph=False,retain_graph=False):
	"""
	compute divergence of x = f(y)
	:x: function outputs
	:y: parameters (expected shape: [batchsize x 3] - 3 for x,y,t)
	"""
	div = grad(x[:,0],y,create_graph=create_graph,retain_graph=True)[:,0:1]+grad(x[:,1],y,create_graph=create_graph,retain_graph=retain_graph)[:,1:2]
	return div

