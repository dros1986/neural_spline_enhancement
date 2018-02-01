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
from test import test
import customTransforms

class cols:
	GREEN = '\033[92m'; BLUE = '\033[94m'; CYAN = '\033[36m';
	LIGHT_GRAY = '\033[37m'; ENDC = '\033[0m'

def showImage(writer, batch, name, n_iter):
	# batch2image
	img = utils.make_grid(batch, nrow=int(math.sqrt(batch.size(0))), padding=3)
	img = torch.clamp(img,0,1)
	writer.add_image(name, img, n_iter)

def plotSplines(writer, splines, name, n_iter):
	# get range
	my_dpi = 100
	r = torch.arange(0,1,1.0/splines.size(2)).numpy()
	splines_images = torch.Tensor([])
	# plot each spline
	for i in range(splines.size(0)):
		plt.figure(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)
		cur_spline = splines[i,:,:]
		# plot splines
		plt.plot(r,cur_spline[0,:].numpy(),color="r", linewidth=4)
		plt.plot(r,cur_spline[1,:].numpy(),color="g", linewidth=4)
		plt.plot(r,cur_spline[2,:].numpy(),color="b", linewidth=4)
		# set range and show grid
		plt.xlim(0,1)
		plt.ylim(0,1)
		plt.grid()
		# save plot to PIL image
		buf = io.BytesIO()
		plt.savefig(buf, format='png', bbox_inches='tight', dpi=my_dpi)
		plt.close()
		buf.seek(0)
		im = Image.open(buf)
		tim = transforms.ToTensor()(im).unsqueeze(0)
		tim = tim[:,:3,:,:]
		if splines_images.ndimension() == 0:
			splines_images = tim
		else:
			splines_images = torch.cat((splines_images,tim),0)
	# plot
	showImage(writer, splines_images, name, n_iter)



def train(dRaw, dExpert, train_list, val_list, batch_size, epochs, npoints, nc, weights_from=''):
		# define summary writer
		expname = 'spline_npoints_{:d}_nfilters_{:d}'.format(npoints,nc)
		writer = SummaryWriter(os.path.join('./logs/', time.strftime('%Y-%m-%d %H:%M:%S'), expname))
		# define transform
		trans = customTransforms.Compose([
					customTransforms.RandomResizedCrop(size=256, scale=(1,1.2),ratio=(0.9,1.1)), \
					customTransforms.RandomHorizontalFlip(), \
					customTransforms.ToTensor(), \
				])
		# create dataloader
		train_data_loader = data.DataLoader(
				Dataset(dRaw, dExpert, train_list, trans=trans, include_filenames=False),
				batch_size = batch_size,
				shuffle = True,
				num_workers = cpu_count(),
				# num_workers = 0,
				drop_last = False
		)
		# create neural spline
		spline = NeuralSpline(npoints,nc).cuda()
		# define optimizer
		optimizer = torch.optim.Adam(spline.parameters(), lr=0.00001)
		# ToDo: load weigths
		start_epoch = 0
		if weights_from:
			state = torch.load(weights_from)
			spline.load_state_dict(state['state_dict'])
			optimizer.load_state_dict(state['optimizer'])
			start_epoch = state['nepoch']
		# for each batch
		curr_iter,best_l2_lab = 0,0
		for nepoch in range(start_epoch, epochs):
			for bn, (raw,expert) in enumerate(train_data_loader):
				#print(bn)
				start_time = time.time()
				# reset gradients
				optimizer.zero_grad()
				# convert to cuda Variable
				raw, expert = Variable(raw.cuda()), Variable(expert.cuda(), requires_grad=False)
				# apply spline transform
				out, splines = spline(raw)
				# calculate loss
				# loss = F.mse_loss(out,expert)
				# out = torch.clamp(out,0,1)
				out_lab, gt_lab = spline.rgb2lab(out), spline.rgb2lab(expert)
				loss = F.mse_loss(out_lab, gt_lab)
				# plot loss
				writer.add_scalar('train_loss', loss.data.cpu().mean(), curr_iter)
				# backprop
				loss.backward()
				# update optimizer
				if bn % (100 if curr_iter < 200 else 200) == 0:
					showImage(writer, raw.data, 'train_input', curr_iter)
					showImage(writer, out.data, 'train_output', curr_iter)
					showImage(writer, expert.data, 'train_gt', curr_iter)
					plotSplines(writer, splines, 'splines', curr_iter)
					# add histograms
					for name, param in spline.named_parameters():
						writer.add_histogram(name, param.clone().cpu().data.numpy(), curr_iter)
				if bn % 100 == 0:
					torch.save({
						'state_dict': spline.state_dict(),
						'optimizer': optimizer.state_dict(),
						'nepoch' : nepoch,
					}, './{}.pth'.format(expname))
				# update weights
				optimizer.step()
				# get time
				elapsed_time = time.time() - start_time
				# define string
				s = \
					( \
					 cols.BLUE + '[{:02d}]' + \
					 cols.BLUE + '[{:03d}/{:3d}]' + \
					 cols.BLUE + '[{:06d}]' + \
					 cols.CYAN  + ' tm: ' + cols.BLUE + '{:.4f}' + \
					 cols.LIGHT_GRAY + ' Loss: ' + cols.GREEN + '{:.4f}' + cols.ENDC \
					).format(nepoch,bn,train_data_loader.__len__(),curr_iter, elapsed_time, loss.data[0])
				print(s)
				# update iter num
				curr_iter = curr_iter + 1
			# at the end of each epoch, test values
			l2_lab, l2_l = test(dRaw, dExpert, val_list, batch_size, spline, outdir='')
			writer.add_scalar('L2-LAB', l2_lab, curr_iter)
			writer.add_scalar('L2-L', l2_l, curr_iter)
			# save best model
			if nepoch == 0 or (nepoch>0 and l2_lab<best_l2_lab):
				best_l2_lab = l2_lab
				torch.save({
					'state_dict': spline.state_dict(),
					'optimizer': optimizer.state_dict(),
					'nepoch' : nepoch,
				}, './{}_best_{:.4f}.pth'.format(expname,l2_lab))
			# print
			print('{}CURR:{} L2_LAB = {}{:.4f}{} - L2_L = {}{:.4f}{}'.format(cols.BLUE,cols.LIGHT_GRAY, cols.GREEN, l2_lab, cols.LIGHT_GRAY, cols.GREEN, l2_l, cols.ENDC))
			print('{}BEST:{} L2_LAB = {}{:.4f}{}'.format(cols.BLUE, cols.LIGHT_GRAY, cols.GREEN, best_l2_lab, cols.ENDC))
