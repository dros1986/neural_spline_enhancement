import os,sys,math,time,io
import torch
import torch.nn as nn
import torch.optim as optim
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
import ptcolor



def test(dRaw, dExpert, test_list, batch_size, spline, deltae=94, apply_to='rgb', dSemSeg='', dSaliency='', \
		nclasses=150, outdir=''):
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
				Dataset(dRaw, dExpert, test_list, dSemSeg, dSaliency, nclasses=nclasses, include_filenames=True),
				batch_size = batch_size,
				shuffle = True,
				num_workers = cpu_count(),
				# num_workers = 0,
				drop_last = False
		)
		# create output mat
		de,diff_l,nimages = [0 for i in range(len(dExpert))],[0 for i in range(len(dExpert))],0
		# calculate differences
		for bn, (images,fns) in enumerate(test_data_loader):
			raw = images[0]
			experts = images[1:]
			nimages += experts[0].size(0)
			# to GPU
			raw = raw.cuda()
			# for i in range(len(experts)):
			# 	experts[i] = Variable(experts[i].cuda(), requires_grad=False)
			# apply spline transform
			out_rgb, splines = spline(raw)
			# get size of images
			h,w = out_rgb[i].size(2),out_rgb[i].size(3)
			# calculate diff
			for i in range(len(out_rgb)):
				out_rgb[i] = out_rgb[i].cpu().data
				# convert to LAB
				gt_lab = spline.rgb2lab(experts[i].cuda())
				if apply_to=='rgb':
					out_rgb[i] = torch.clamp(out_rgb[i],0,1)
					out_lab = spline.rgb2lab(out_rgb[i].cuda())
				else:
					out_lab = out_rgb[i]
				# calculate deltaE
				if deltae == 94:
					cur_de = ptcolor.deltaE94(out_lab, gt_lab)
				else:
					cur_de = ptcolor.deltaE(out_lab, gt_lab)
				# add current deltaE to accumulator
				de[i] += cur_de.sum() #.mean()
				# calculate L1 on L channel and add to
				diff_l[i] += torch.abs(out_lab[:,0,:,:]-gt_lab[:,0,:,:]).sum() #.mean()
				# save if required
				if outdir:
					# convert if required
					if not apply_to=='rgb':
						for j in range(out_rgb[i].size(0)):
							out_rgb[i] = spline.lab2rgb(out_rgb[i])
							out_rgb[i] = torch.clamp(out_rgb[i],0,1)
					# save each image
					for j in range(out_rgb[i].size(0)):
						cur_fn = fns[j]
						cur_img = out_rgb[i][j,:,:,:].cpu().numpy().transpose((1,2,0))
						cur_img = (cur_img*255).astype(np.uint8)
						cur_img = Image.fromarray(cur_img)
						cur_img.save(os.path.join(outdir,experts_names[i],cur_fn))
		# calculate differences
		for i in range(len(de)):
			de[i] /= nimages*h*w
			diff_l[i] /= nimages*h*w

		return de, diff_l
