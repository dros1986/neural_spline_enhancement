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
		diff_lab,diff_l,nimages = 0,0,0
		# calculate differences
		for bn, (raw,expert,fns) in enumerate(test_data_loader):
			# convert to cuda Variable
			raw = Variable(raw.cuda())
			# apply spline transform
			out_rgb, splines = spline(raw)
			out_rgb = out_rgb.cpu().data
			# set bounds
			out_rgb = torch.clamp(out_rgb,0,1)
			# convert to LAB
			out_lab, gt_lab = spline.rgb2lab(out_rgb.cuda()), spline.rgb2lab(expert.cuda())
			# calculate diff
			cur_diff = torch.pow((out_lab-gt_lab),2)
			cur_diff_lab = torch.sqrt(cur_diff)
			cur_diff_l = torch.sqrt(cur_diff[:,0,:,:])
			diff_lab = cur_diff_lab if bn==0 else diff_lab+cur_diff_lab
			diff_l = cur_diff_l if bn==0 else diff_l+cur_diff_l
			nimages += out_lab.size(0)
			# save if required
			if outdir:
				# save each image
				for i in range(out_rgb.size(0)):
					cur_fn = fns[i]
					cur_img = out_rgb[i,:,:,:].cpu().numpy().transpose((1,2,0))
					cur_img = (cur_img*255).astype(np.uint8)
					cur_img = Image.fromarray(cur_img)
					cur_img.save(os.path.join(outdir,cur_fn))
		# calculate differences
		l2_lab = diff_lab.sum() / (nimages*diff_lab.size(1)*diff_lab.size(2)*diff_lab.size(3))
		l2_l = diff_l.sum() / (nimages*diff_l.size(1)*diff_l.size(2))
		return l2_lab, l2_l
