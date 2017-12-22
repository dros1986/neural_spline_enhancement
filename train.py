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

class cols:
	GREEN = '\033[92m'; BLUE = '\033[94m'; CYAN = '\033[36m';
	LIGHT_GRAY = '\033[37m'; ENDC = '\033[0m'

def showImage(writer, batch, name):
	# convert image to numpy
	img = utils.make_grid(batch, nrow=int(math.sqrt(batch.size(0))), padding=3)
	img = img.cpu().numpy().transpose((1, 2, 0))
	img = (img*255).astype(np.uint8)
	writer.add_image(name, img)

def plotSplines(writer, splines, name):
	# get range
	my_dpi = 100
	r = torch.arange(0,1,1.0/splines.size(1)).numpy()
	splines_images = torch.Tensor([])
	# plot each spline
	for i in range(splines.size(0)):
		plt.figure(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)
		cur_spline = splines[i,:]
		# plot spline
		plt.plot(r,cur_spline.numpy(),linewidth=4)
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
	showImage(writer, splines_images, name)



def train(dRaw, dExpert, train_list, val_list, batch_size, epochs, npoints, weights_from=''):
		# define summary writer
		writer = SummaryWriter(os.path.join('./logs/', time.strftime('%Y-%m-%d %H:%M:%S'), 'experiment'))
		# create dataloader
		train_data_loader = data.DataLoader(
				Dataset(dRaw, dExpert, train_list, include_filenames=False),
				batch_size = batch_size,
				shuffle = True,
				#num_workers = cpu_count(),
				num_workers = 0,
				drop_last = False
		)
		# create neural spline
		nf = 100
		spline = NeuralSpline(npoints,nf).cuda()
		# define optimizer
		optimizer = torch.optim.Adam(spline.parameters(), lr=0.001)
		# ToDo: load weigths
		start_epoch = 0
		if weights_from:
			state = torch.load(weights_from)
			spline.load_state_dict(state['state_dict'])
			optimizer.load_state_dict(state['optimizer'])
			start_epoch = state['nepoch']
		# for each batch
		curr_iter = 0
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
				loss = F.l1_loss(out,expert)
				# plot loss
				writer.add_scalar('train_loss', loss.data.cpu().mean(), curr_iter)
				# backprop
				loss.backward()
				# update optimizer
				if bn % (10 if nepoch < 3 else 100) == 0:
					showImage(writer, raw.data, 'train_input')
					showImage(writer, out.data, 'train_output')
					showImage(writer, expert.data, 'train_gt')
					plotSplines(writer, splines, 'splines')
				if bn % 100 == 0:
					torch.save({
						'state_dict': spline.state_dict(),
						'optimizer': optimizer.state_dict(),
						'nepoch' : nepoch,
					}, './checkpoint.pth')
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
					).format(nepoch,bn,curr_iter,train_data_loader.__len__(), elapsed_time, loss.data[0])
				print(s)
				# update iter num
				curr_iter = curr_iter + 1
