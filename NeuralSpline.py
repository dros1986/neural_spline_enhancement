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
		self.c2 = nn.Conv2d(nc, 2*nc, kernel_size=3, stride=2, padding=0)
		self.b2 = nn.BatchNorm2d(2*nc)
		self.c3 = nn.Conv2d(2*nc, 4*nc, kernel_size=3, stride=2, padding=0)
		self.b3 = nn.BatchNorm2d(4*nc)
		self.c4 = nn.Conv2d(4*nc, 8*nc, kernel_size=3, stride=2, padding=0)
		self.b4 = nn.BatchNorm2d(8*nc)
		self.c5 = nn.Conv2d(8*nc, 16*nc, kernel_size=3, stride=2, padding=0)
		self.b5 = nn.BatchNorm2d(16*nc)

		self.l1 = nn.Linear(16*nc*7*7, 100*n)
		self.l2 = nn.Linear(100*n, 3*n)

	def rgb2lab(self,x):
		M,N = x.size(2),x.size(3)
		R,G,B = x[:,0,:,:].contiguous(),x[:,1,:,:].contiguous(),x[:,2,:,:].contiguous()
		R,G,B = R.view(x.size(0),1,-1),G.view(x.size(0),1,-1),B.view(x.size(0),1,-1)
		RGB = torch.cat((R,G,B),1)
		# RGB ProPhoto -> CIELAB XYZ
		MAT = [[0.7976749, 0.1351917, 0.0313534], \
		       [0.2880402, 0.7118741, 0.0000857], \
		       [0.0000000, 0.0000000, 0.8252100]]
		MAT = torch.Tensor(MAT)
		MAT = MAT.unsqueeze(0).repeat(RGB.size(0),1,1)
		if isinstance(RGB,Variable): MAT = Variable(MAT,requires_grad=False)
		if RGB.is_cuda: MAT = MAT.cuda()
		XYZ = torch.bmm(MAT,RGB)
		# Normalize for D65 white point
		X = XYZ[:,0,:]/0.950456
		Y = XYZ[:,1,:]
		Z = XYZ[:,2,:]/1.088754
		T = 0.008856
		XT,YT,ZT = X>T, Y>T, Z>T
		XT,YT,ZT = XT.float(), YT.float(), ZT.float()
		mn = Variable(torch.Tensor([T]).cuda(), requires_grad=False)
		Y3 = torch.max(Y,mn)**(1.0/3)
		fX = XT * torch.max(X,mn)**(1.0/3) + (1-XT) * (7.787*X + 16.0/116)
		fY = YT * Y3 + (1-YT) * (7.787 * Y + 16.0/116)
		fZ = ZT * torch.max(Z,mn)**(1.0/3) + (1-ZT) * (7.787*Z + 16.0/116)
		# debug
		# if np.isnan(np.sum(fZ.data.cpu().numpy())):
		# 	print('ERRORE')
		# get LAB channels
		L = YT * (116.0 * Y3 - 16.0) + (1-YT) * (903.3 * Y)
		a = 500 * (fX - fY)
		b = 200 * (fY - fZ)
		L,a,b = L.view(-1,1,M,N), a.view(-1,1,M,N), b.view(-1,1,M,N)
		# return
		LAB = torch.cat((L,a,b),1)
		return LAB
		# PROVA
		# X,Y,Z = X.contiguous(),Y.contiguous(),Z.contiguous()
		# X,Y,Z = X.view(-1,1,M,N), Y.view(-1,1,M,N), Z.view(-1,1,M,N)
		# XYZ = torch.cat((X,Y,Z),1)
		# return XYZ


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
		ys = ys.view(ys.size(0),3,-1)
		ys /= 100
		# now we got xs and ys. We need to create the interpolating spline
		out = Variable(torch.zeros(batch.size())).cuda()
		vals = Variable(torch.arange(0,1,1.0/255),requires_grad=False).cuda()
		splines = torch.zeros(batch.size(0),3,vals.size(0))
		# for each image
		for nimg in range(batch.size(0)):
			# get image and corresponding ys
			# cur_img = batch[nimg,:,:,:]
			# for each channel
			for ch in range(3):
				# get current channel
				cur_ch = batch[nimg,ch,:,:]
				cur_ys = ys[nimg,ch,:]
				# interpolate spline with found ys
				identity = torch.arange(0,cur_ys.size(0))/(cur_ys.size(0)-1)
				identity = Variable(identity,requires_grad=False).cuda()
				cur_coeffs = self.interpolate(cur_ys+identity)
				new_ch = self.apply(cur_coeffs, cur_ch.view(-1))
				splines[nimg,ch,:] = self.apply(cur_coeffs,vals).data.cpu()
				out[nimg,ch,:,:] = new_ch
		# return
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
