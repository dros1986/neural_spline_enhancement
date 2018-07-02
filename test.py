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



def test(dRaw, dExpert, test_list, batch_size, spline, deltae=94, dSemSeg='', dSaliency='', \
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
				# num_workers = cpu_count(),
				num_workers = 0,
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
			# compute transform
			out, splines = spline(raw)
			# detach all
			out = [e.detach() for e in out]
			# get size of images
			h,w = out[i].size(2),out[i].size(3)
			# for each expert
			for i in range(len(out)):
				# convert gt and output in lab (remember that spline in test/lab converts back in rgb)
				gt_lab = spline.rgb2lab(experts[i].cuda())
				ot_lab = spline.rgb2lab(out[i].cuda())
				# calculate deltaE
				if deltae == 94:
					cur_de = ptcolor.deltaE94(ot_lab, gt_lab)
				else:
					cur_de = ptcolor.deltaE(ot_lab, gt_lab)
				# add current deltaE to accumulator
				de[i] += cur_de.sum()
				# calculate L1 on L channel and add to
				diff_l[i] += torch.abs(ot_lab[:,0,:,:]-gt_lab[:,0,:,:]).sum()
				# save if required
				if outdir:
					# save each image
					for j in range(out[i].size(0)):
						cur_fn = fns[j]
						cur_img = out[i][j,:,:,:].cpu().numpy().transpose((1,2,0))
						cur_img = (cur_img*255).astype(np.uint8)
						cur_img = Image.fromarray(cur_img)
						cur_img.save(os.path.join(outdir,experts_names[i],cur_fn))
		# calculate differences
		for i in range(len(de)):
			de[i] /= nimages*h*w
			diff_l[i] /= nimages*h*w
		# return values
		return de, diff_l
