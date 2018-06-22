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



class NeuralSpline(nn.Module):
	def __init__(self, n, nc, nexperts, downsample_strategy='avgpool'):
		super(NeuralSpline, self).__init__()
		# define class params
		self.n = n
		self.x0 = 0
		self.step = 1.0 / (n-1.0)
		self.nexperts = nexperts
		momentum = 0.01
		# compute interpolation matrix (will be stored in self.matrix)
		self._precalc()
		# define net layers
		self.c1 = nn.Conv2d(3, nc, kernel_size=3, stride=2, padding=0)
		self.c2 = nn.Conv2d(nc, 2*nc, kernel_size=3, stride=2, padding=0)
		self.b2 = nn.BatchNorm2d(2*nc, momentum=momentum)
		self.c3 = nn.Conv2d(2*nc, 4*nc, kernel_size=3, stride=2, padding=0)
		self.b3 = nn.BatchNorm2d(4*nc, momentum=momentum)
		self.c4 = nn.Conv2d(4*nc, 8*nc, kernel_size=3, stride=2, padding=0)
		self.b4 = nn.BatchNorm2d(8*nc, momentum=momentum)
		self.c5 = nn.Conv2d(8*nc, 16*nc, kernel_size=3, stride=2, padding=0)
		self.b5 = nn.BatchNorm2d(16*nc, momentum=momentum)
		# define downsample layers
		if downsample_strategy=='maxpool':
			self.downsample = nn.MaxPool2d(7, stride=1)
			self.l1 = nn.Linear(16*nc, 16*nc)
			self.l2 = nn.Linear(16*nc, 3*n*self.nexperts)
		elif downsample_strategy=='avgpool':
			self.downsample = nn.AvgPool2d(7, stride=1)
			self.l1 = nn.Linear(16*nc, 16*nc)
			self.l2 = nn.Linear(16*nc, 3*n*self.nexperts)
		else:
			self.downsample = nn.Sequential(
				nn.Conv2d(16*nc, 32*nc, kernel_size=3, stride=2, padding=0),
				nn.BatchNorm2d(32*nc, momentum=momentum),
				nn.ReLU(True),
				nn.Conv2d(32*nc, 64*nc, kernel_size=3, stride=2, padding=0),
				nn.BatchNorm2d(64*nc, momentum=momentum),
				nn.ReLU(True),
			)
			self.l1 = nn.Linear(64*nc, 32*nc)
			self.l2 = nn.Linear(32*nc, 3*n*self.nexperts)

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


	def forward(self, batch):
		# get xs of the points with CNN
		ys = F.relu(self.c1(batch))
		ys = self.b2(F.relu(self.c2(ys)))
		ys = self.b3(F.relu(self.c3(ys)))
		ys = self.b4(F.relu(self.c4(ys)))
		ys = self.b5(F.relu(self.c5(ys)))
		ys = self.downsample(ys)
		ys = ys.view(ys.size(0),-1)
		ys = F.relu(self.l1(ys))
		ys = self.l2(ys)
		ys = ys.view(ys.size(0),self.nexperts,3,-1)
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


def unique(tensor1d):
    t, idx = np.unique(tensor1d.numpy(), return_inverse=True)
    return torch.from_numpy(t), torch.from_numpy(idx)

if __name__ == "__main__":
	n = 10
	nf = 100
	spline = NeuralSpline(n,nf)
	spline.cuda()
	# img = torch.rand(1,3,256,256)
	# px_vals = unique(img)
	img = Variable(torch.rand(5,3,256,256)).cuda()
	out, splines = spline(img)
	print(out.size())
	import ipdb; ipdb.set_trace()
	#ris = spline(img)
