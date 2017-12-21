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



class NeuralSpline(nn.Module):
	def __init__(self, n, nc):
		super(NeuralSpline, self).__init__()
		# define class params
		self.n = n
		self.x0 = 0
		self.step = 1.0 / (n-1.0)
		# compute interpolation matrix (will be stored in self.matrix)
		self._precalc()
		# define net layers
		self.c1 = nn.Conv2d(3, nc, kernel_size=3, stride=2, padding=0)
		self.b1 = nn.BatchNorm2d(nc)
		self.c2 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)
		self.b2 = nn.BatchNorm2d(nc)
		self.c3 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)
		self.b3 = nn.BatchNorm2d(nc)
		self.c4 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)
		self.b4 = nn.BatchNorm2d(nc)
		self.c5 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)
		self.b5 = nn.BatchNorm2d(nc)

		self.l1 = nn.Linear(nc*7*7, 2*n)
		self.l2 = nn.Linear(2*n, n)


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
		ones = Variable(torch.zeros(xf.size()),requires_grad=False).cuda()
		ex = torch.stack([xf ** 3, xf ** 2, xf, ones], dim=0)
		#y = np.dot(coeffs.transpose(0,1), ex)
		y = torch.mm(coeffs.transpose(0,1), ex)
		#xi = xi.long()
		# create constant mat
		ohe = OneHotEncoder(n_values=y.size(0), categorical_features='all', dtype=np.float64, sparse=False, handle_unknown='error')
		sel_mat = ohe.fit_transform(xi.data.cpu().numpy().reshape(-1,1))
		sel_mat = Variable(torch.from_numpy(sel_mat), requires_grad=False)
		sel_mat = sel_mat.transpose(0,1).float().cuda()
		# multiply to get the right coeffs
		res = y*sel_mat
		res = res.sum(0)
		# return
		return res


	def forward(self, batch):
		# get xs of the points with CNN
		ys = self.b1(F.relu(self.c1(batch)))
		ys = self.b2(F.relu(self.c2(ys)))
		ys = self.b3(F.relu(self.c3(ys)))
		ys = self.b4(F.relu(self.c4(ys)))
		ys = self.b5(F.relu(self.c5(ys)))
		ys = ys.view(ys.size(0),-1)
		ys = F.relu(self.l1(ys))
		ys = self.l2(ys)
		# now we got xs and ys. We need to create the interpolating spline
		out = Variable(torch.zeros(batch.size())).cuda()
		# for each image
		for nimg in range(batch.size(0)):
			# interpolate spline with found ys
			cur_img = batch[nimg,:,:,:]
			cur_ys = ys[nimg,:]
			cur_coeffs = self.interpolate(cur_ys)
			new_img = self.apply(cur_coeffs, cur_img.view(-1))
			out[nimg] = new_img
		# return
		return out


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
	out = spline(img)
	print(out.size())
	import ipdb; ipdb.set_trace()
	#ris = spline(img)
