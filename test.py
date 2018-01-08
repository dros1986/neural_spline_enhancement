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
from tensorboard import SummaryWriter
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def test(dRaw, dExpert, test_list, batch_size, npoints, nc, weights_file, out_dir):
	# create out_dir if not existing
	if not os.path.isdir(out_dir): os.makedirs(out_dir)
	# create net
	spline = NeuralSpline(npoints,nc).cuda()
	# create dataloader
	test_data_loader = data.DataLoader(
			Dataset(dRaw, dExpert, test_list, include_filenames=True),
			batch_size = batch_size,
			shuffle = True,
			num_workers = cpu_count(),
			# num_workers = 0,
			drop_last = False
	)
	# load weights
	state = torch.load(weights_file)
	spline.load_state_dict(state['state_dict'])
	# for each batch
	for bn, (raw,expert,fns) in enumerate(test_data_loader):
		# convert to cuda Variable
		raw, expert = Variable(raw.cuda()), Variable(expert.cuda(), requires_grad=False)
		# apply spline transform
		out, splines = spline(raw)
		# set bounds
		out = torch.clamp(out,0,1)
		# save each image
		for i in range(out.size(0)):
			cur_fn = fns[i]
			cur_img = out[i,:,:,:].data.cpu().numpy().transpose((1,2,0))
			cur_img = (cur_img*255).astype(np.uint8)
			cur_img = Image.fromarray(cur_img)
			cur_img.save(os.path.join(out_dir,cur_fn))
