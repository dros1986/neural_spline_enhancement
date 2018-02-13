import os,sys,math,time,io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, utils
from Dataset import Dataset
from NeuralSpline import NeuralSpline
from tensorboardX import SummaryWriter
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def test(dRaw, dExpert, test_list, batch_size, spline, outdir=''):
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
			# calculate diff
			for i in range(len(out_rgb)):
				out_rgb[i] = out_rgb[i].cpu().data
				# set bounds
				out_rgb[i] = torch.clamp(out_rgb[i],0,1)
				# convert to LAB
				out_lab, gt_lab = spline.rgb2lab(out_rgb[i].cuda()), spline.rgb2lab(experts[i].cuda())
				# calculate diff # sqrt(sum_c((out_chw,gt_chw)^2))
				cur_diff = torch.pow((out_lab-gt_lab),2)
				cur_diff_lab = torch.sqrt(torch.sum(cur_diff,1)) # manca la somma
				cur_diff_l = torch.sqrt(cur_diff[:,0,:,:])
				diff_lab[i] = cur_diff_lab if bn==0 else diff_lab[i]+cur_diff_lab
				diff_l[i] = cur_diff_l if bn==0 else diff_l[i]+cur_diff_l
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
		l2_lab,l2_l = [],[]
		for i in range(len(diff_lab)):
			#l2_lab.append(diff_lab[i].sum() / (nimages*diff_lab[i].size(1)*diff_lab[i].size(2)*diff_lab[i].size(3)))
			l2_lab.append(diff_lab[i].sum() / (nimages*diff_lab[i].size(1)*diff_lab[i].size(2)))
			l2_l.append(diff_l[i].sum() / (nimages*diff_l[i].size(1)*diff_l[i].size(2)))

		return l2_lab, l2_l
