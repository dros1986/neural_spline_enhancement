import os,sys,math,time,io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, utils
from Dataset import Dataset
from NeuralSpline import NeuralSpline, HDRNet, Baseline
from tensorboardX import SummaryWriter
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import ptcolor



def test(dRaw, dExpert, test_list, batch_size, spline, outdir=''):
		spline.eval()
		# create folder
		if outdir and not os.path.isdir(outdir): os.makedirs(outdir)
		# get experts names and create corresponding folder
		experts_names = []
		for i in range(len(dExpert)):
			experts_names.append([s for s in dExpert[i].split(os.sep) if s][-1])
			if outdir and not os.path.isdir(os.path.join(outdir,experts_names[-1])):
				os.makedirs(os.path.join(outdir,experts_names[-1]))
		# create dataloader
		test_data_loader = data.DataLoader(
				Dataset(dRaw, dExpert, test_list, include_filenames=True),
				batch_size = batch_size,
				shuffle = True,
				num_workers = cpu_count(),
				# num_workers = 0,
				drop_last = False
		)
		# create output mat
		diff_lab,diff_l,nimages = [0 for i in range(len(dExpert))],[0 for i in range(len(dExpert))],0
		diff_lab94 = diff_lab[:]
		# calculate differences
		for bn, (images,fns) in enumerate(test_data_loader):
			raw = images[0]
			experts = images[1:]
			nimages += experts[0].size(0)
			# convert to cuda variable
			raw = Variable(raw.cuda())
			# for i in range(len(experts)):
			# 	experts[i] = Variable(experts[i].cuda(), requires_grad=False)
			# apply spline transform
			out_rgb, splines = spline(raw)
			# get size of images
			h,w = out_rgb[i].size(2),out_rgb[i].size(3)
			# calculate diff
			for i in range(len(out_rgb)):
				out_rgb[i] = out_rgb[i].cpu().data
				# set bounds
				out_rgb[i] = torch.clamp(out_rgb[i],0,1)
				# convert to LAB
				out_lab, gt_lab = spline.rgb2lab(out_rgb[i].cuda()), spline.rgb2lab(experts[i].cuda())
				# calculate diff # sqrt(sum_c((out_chw-gt_chw)^2))
				cur_diff = torch.pow((out_lab-gt_lab),2)         # 10 3 256 256
				# cur_diff_lab = torch.sqrt(torch.sum(cur_diff,1)) # 10 256 256
				cur_diff_lab94 = ptcolor.deltaE94(gt_lab, out_lab).squeeze(1)  # !!!
				cur_diff_lab = ptcolor.deltaE(gt_lab, out_lab).squeeze(1)  # !!!
				cur_diff_l = torch.sqrt(cur_diff[:,0,:,:])       # 10 256 256
				# sum all
				diff_lab[i] += cur_diff_lab.sum()
				diff_lab94[i] += cur_diff_lab94.sum()
				diff_l[i] += cur_diff_l.sum()
				# save if required
				if outdir:
					# save each image
					for j in range(out_rgb[i].size(0)):
						cur_fn = fns[j]
						cur_img = out_rgb[i][j,:,:,:].cpu().numpy().transpose((1,2,0))
						cur_img = (cur_img*255).astype(np.uint8)
						cur_img = Image.fromarray(cur_img)
						cur_img.save(os.path.join(outdir,experts_names[i],cur_fn))
		# calculate differences
		for i in range(len(diff_lab)):
			diff_lab[i] /= nimages*h*w
			diff_lab94[i] /= nimages*h*w
			diff_l[i] /= nimages*h*w

		return diff_lab, diff_lab94, diff_l
