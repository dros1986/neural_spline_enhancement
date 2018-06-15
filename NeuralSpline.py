import os,sys, math, argparse
import torch
import torch.nn as nn
import torch.nn.init as winit
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import torch.utils.data as data
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import ptcolor


def linear_sRGB(rgb):
        T = 0.04045
        c = (rgb < T).float()
        return c * rgb / 12.92 + (1 - c) * torch.pow(torch.abs(rgb + 0.055) / 1.055, 2.4)


class NeuralSplineBase(nn.Module):
	def __init__(self, n, nc, nexperts):
		super().__init__()
		# define class params
		self.n = n
		self.x0 = 0
		self.step = 1.0 / (n-1.0)
		self.nexperts = nexperts
		# compute interpolation matrix (will be stored in self.matrix)
		self._precalc()

	def rgb2lab(self,x, from_space='srgb'):
		return ptcolor.rgb2lab(x, clip_rgb=not self.training, gamma_correction=True)

	def _precalc(self):
		""" Calculate interpolation mat for finding Ms.
			It will be stored in self.matrix.
		"""
		n = self.n
		mat = 4 * np.eye(n - 2)
		np.fill_diagonal(mat[1:, :-1], 1)
		np.fill_diagonal(mat[:-1, 1:], 1)
		A = 6 * np.linalg.inv(mat) / (self.step ** 2)
		z = np.zeros(n - 2)
		A = np.vstack([z, A, z])

		B = np.zeros([n - 2, n])
		np.fill_diagonal(B, 1)
		np.fill_diagonal(B[:, 1:], -2)
		np.fill_diagonal(B[:, 2:], 1)
		self.matrix = np.dot(A, B)
		self.matrix = torch.from_numpy(self.matrix).float()
		self.matrix = Variable(self.matrix,requires_grad=False).cuda()

	def interpolate(self, ys):
		""" compute the coefficients of the polynomials
		"""
		# get coefficients of the polinomials that compose the spline
		h = self.step
		#M = self.matrix.dot(ys)
		M = torch.mm(self.matrix,ys.view(-1,1)).squeeze()
		a = (M[1:] - M[:-1]) / (6 * h)
		b = M[:-1] / 2
		c = (ys[1:] - ys[:-1]) / h - (M[1:] + 2 * M[:-1]) * (h / 6)
		d = ys[:-1]
		coeffs = torch.stack([a,b,c,d], dim=0)
		return coeffs

	def apply(self, coeffs, x):
		""" interpolate new data using coefficients
		"""
		xi = torch.clamp((x - self.x0) / self.step, 0, self.n-2)
		xi = torch.floor(xi)
		xf = x - self.x0 - xi*self.step
		ones = Variable(torch.ones(xf.size()),requires_grad=False).cuda()
		ex = torch.stack([xf ** 3, xf ** 2, xf, ones], dim=0)
		#y = np.dot(coeffs.transpose(0,1), ex)
		y = torch.mm(coeffs.transpose(0,1), ex)
		# create constant mat
		sel_mat = torch.zeros(y.size(0),xi.size(0)).cuda()
		rng = torch.arange(0,xi.size(0)).cuda()
		sel_mat[xi.data.long(),rng.long()]=1
		sel_mat = Variable(sel_mat, requires_grad=False)
		# multiply to get the right coeffs
		res = y*sel_mat
		res = res.sum(0)
		# return
		return res

	def enhanceImage(self, input_image, ys):
		image = input_image.clone()
		vals = Variable(torch.arange(0,1,1.0/255),requires_grad=False).cuda()
		splines = torch.zeros(3,vals.size(0))
		# for each channel of the image, define spline and apply it
		for ch in range(image.size(0)):
			cur_ch = image[ch,:,:].clone()
			cur_ys = ys[ch,:].clone()
			# calculate spline upon identity + found ys
			identity = torch.arange(0,cur_ys.size(0))/(cur_ys.size(0)-1)
			identity = Variable(identity,requires_grad=False).cuda()
			cur_coeffs = self.interpolate(cur_ys+identity)
			image[ch,:,:] = self.apply(cur_coeffs, cur_ch.view(-1)).view(cur_ch.size())
			splines[ch,:] = self.apply(cur_coeffs,vals).data.cpu()
		return image, splines

	def _final_proc(self, batch, ys):
                # now we got xs and ys. We need to create the interpolating spline
		vals = torch.arange(0,1,1.0/255)
		out = [Variable(torch.zeros(batch.size())).cuda() for i in range(self.nexperts)]
		splines = [torch.zeros(batch.size(0),3,vals.size(0)) for i in range(self.nexperts)]
		# for each expert
		for nexp in range(self.nexperts):
			# init output vars
			cur_out = Variable(torch.zeros(batch.size())).cuda()
			cur_vals = Variable(torch.arange(0,1,1.0/255),requires_grad=False).cuda()
			cur_splines = torch.zeros(batch.size(0),3,vals.size(0))
			# enhance each image with the expert spline
			for nimg in range(batch.size(0)):
				cur_img = batch[nimg,:,:,:]
				cur_ys = ys[nimg,nexp,:,:]
				out[nexp][nimg,:,:,:], splines[nexp][nimg,:,:] = self.enhanceImage(cur_img, cur_ys)
		return out, splines


class NeuralSpline(NeuralSplineBase):
	def __init__(self, n, nc, nexperts):
		super().__init__(n, nc, nexperts)
		momentum = 0.01
		# define net layers
		self.c1 = nn.Conv2d(3, nc, kernel_size=3, stride=2, padding=0)
		self.b1 = nn.BatchNorm2d(nc, momentum=momentum)
		self.c2 = nn.Conv2d(nc, 2*nc, kernel_size=3, stride=2, padding=0)
		self.b2 = nn.BatchNorm2d(2*nc, momentum=momentum)
		self.c3 = nn.Conv2d(2*nc, 4*nc, kernel_size=3, stride=2, padding=0)
		self.b3 = nn.BatchNorm2d(4*nc, momentum=momentum)
		self.c4 = nn.Conv2d(4*nc, 8*nc, kernel_size=3, stride=2, padding=0)
		self.b4 = nn.BatchNorm2d(8*nc, momentum=momentum)
		self.c5 = nn.Conv2d(8*nc, 16*nc, kernel_size=3, stride=2, padding=0)
		self.b5 = nn.BatchNorm2d(16*nc, momentum=momentum)

		self.l1 = nn.Linear(16*nc*1*1, 100*n)  # was 7*7
		self.l2 = nn.Linear(100*n, 3*n*self.nexperts)
        
	def forward(self, batch):
		# get xs of the points with CNN
                ys = F.avg_pool2d(batch, 4)
		# ys = F.relu(self.c1(ys))
		# ys = F.relu(self.c2(ys))
                ys = self.b1(F.relu(self.c1(ys)))
                ys = self.b2(F.relu(self.c2(ys)))
                ys = self.b3(F.relu(self.c3(ys)))
                ys = self.b4(F.relu(self.c4(ys)))
                ys = self.b5(F.relu(self.c5(ys)))
                ys = ys.view(ys.size(0),-1)
                ys = F.relu(self.l1(ys))
                ys = self.l2(ys)
                ys = ys.view(ys.size(0),self.nexperts,3,-1)
                ys /= 100
                return self._final_proc(batch, ys)


class Baseline(NeuralSplineBase):
	def __init__(self, n, nc, nexperts):
		super().__init__(n, nc, nexperts)
		momentum = 0.01
		# define net layers
		self.c1 = nn.Conv2d(3, nc, kernel_size=3, stride=2, padding=0)
		# self.b1 = nn.BatchNorm2d(nc, momentum=momentum)
		self.c2 = nn.Conv2d(nc, 2*nc, kernel_size=3, stride=2, padding=0)
		self.b2 = nn.BatchNorm2d(2*nc, momentum=momentum)
		self.c3 = nn.Conv2d(2*nc, 4*nc, kernel_size=3, stride=2, padding=0)
		self.b3 = nn.BatchNorm2d(4*nc, momentum=momentum)
		self.c4 = nn.Conv2d(4*nc, 8*nc, kernel_size=3, stride=2, padding=0)
		self.b4 = nn.BatchNorm2d(8*nc, momentum=momentum)
		self.c5 = nn.Conv2d(8*nc, 16*nc, kernel_size=3, stride=2, padding=0)
		self.b5 = nn.BatchNorm2d(16*nc, momentum=momentum)
		# self.c6 = nn.Conv2d(16*nc, 32*nc, kernel_size=3, stride=2, padding=0)
		# self.b6 = nn.BatchNorm2d(32*nc, momentum=momentum)
		# self.c7 = nn.Conv2d(32*nc, 64*nc, kernel_size=3, stride=2, padding=0)
		# self.b7 = nn.BatchNorm2d(64*nc, momentum=momentum)
		self.l1 = nn.Linear(16*nc, 16*nc)
		self.l2 = nn.Linear(16*nc, 3*n*self.nexperts)
        
	def forward(self, batch):
		# get xs of the points with CNN
		ys = batch
		ys = F.relu(self.c1(ys))
		ys = self.b2(F.relu(self.c2(ys)))
		ys = self.b3(F.relu(self.c3(ys)))
		ys = self.b4(F.relu(self.c4(ys)))
		ys = self.b5(F.relu(self.c5(ys)))
		# ys = self.b6(F.relu(self.c6(ys)))
		# ys = self.b7(F.relu(self.c7(ys)))
		ys = F.avg_pool2d(ys, ys.size(2))
		ys = ys.view(ys.size(0),-1)
		ys = F.dropout(ys, training=self.training)
		ys = F.relu(self.l1(ys))
		ys = F.dropout(ys, training=self.training)
		ys = self.l2(ys)
		ys = ys.view(ys.size(0),self.nexperts,3,-1)
		# a, b = self._final_proc(batch, ys)  # !!!
		# return [F.sigmoid(a[0])], b              # !!!  
		return self._final_proc(batch, ys)  # !!!        


class HDRNet(NeuralSplineBase):
	"""Replica of the HDRNet from Gharbi et al. (Global path only)."""
	def __init__(self, n, nc, nexperts):
		super().__init__(n, nc, nexperts)
		momentum = 0.01   # Tensorflow default is 0.001
		# define net layers
                # "Splat"
		self.c1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
		self.c2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
		self.b2 = nn.BatchNorm2d(16, momentum=momentum)
		self.c3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
		self.b3 = nn.BatchNorm2d(32, momentum=momentum)
		self.c4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
		self.b4 = nn.BatchNorm2d(64, momentum=momentum)
		self.c5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
		self.b5 = nn.BatchNorm2d(64, momentum=momentum)
		self.c6 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
		self.b6 = nn.BatchNorm2d(64, momentum=momentum)
                # "Global"
		self.l1 = nn.Linear(4 * 4 * 64, 256)
		self.b7 = nn.BatchNorm1d(256, momentum=momentum)
		self.l2 = nn.Linear(256, 128)
		self.b8 = nn.BatchNorm1d(128, momentum=momentum)
		self.l3 = nn.Linear(128, 64)
                # Final map
		self.l4 = nn.Linear(64, 3 * n *self.nexperts)
        
	def forward(self, batch):
                ys = batch
                ys = F.relu(self.c1(ys))  # First layer without BN
                ys = F.relu(self.b2(self.c2(ys)))
                ys = F.relu(self.b3(self.c3(ys)))
                ys = F.relu(self.b4(self.c4(ys)))
                ys = F.relu(self.b5(self.c5(ys)))
                ys = F.relu(self.b6(self.c6(ys)))
                ys = ys.view(ys.size(0),-1)
                ys = F.relu(self.b7(self.l1(ys)))
                ys = F.relu(self.b8(self.l2(ys)))
                ys = F.relu(self.l3(ys))
                ys = self.l4(ys)
                ys = ys.view(ys.size(0),self.nexperts,3,-1)
                ys /= 100
                return self._final_proc(batch, ys)
        

def unique(tensor1d):
    t, idx = np.unique(tensor1d.numpy(), return_inverse=True)
    return torch.from_numpy(t), torch.from_numpy(idx)

if __name__ == "__main__":
	n = 10
	nf = 100
	# spline = NeuralSpline(n, nf, 1)
	spline = HDRNet(n, nf, 1)
	spline.cuda()
	# img = torch.rand(1,3,256,256)
	# px_vals = unique(img)
	img = Variable(torch.rand(5,3,256,256)).cuda()
	out, splines = spline(img)
	print(out[0].size())
	#import ipdb; ipdb.set_trace()
	#ris = spline(img)
