import os,sys,math,time,io,argparse
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from test import test
import customTransforms
import ptcolor

class cols:
	GREEN = '\033[92m'; BLUE = '\033[94m'; CYAN = '\033[36m';
	LIGHT_GRAY = '\033[37m'; ENDC = '\033[0m'

def showImage(writer, batch, name, n_iter):
	# batch2image
	img = utils.make_grid(batch[:,:3,:,:], nrow=int(math.sqrt(batch.size(0))), padding=3)
	img = torch.clamp(img,0,1)
	writer.add_image(name, img, n_iter)
	# visualize first image maps if any
	if batch.size(1)>3:
		img = batch[0,3:,:,:].unsqueeze(1)
		img = utils.make_grid(img, nrow=int(math.sqrt(batch.size(0))), padding=3)
		img = torch.clamp(img,0,1)
	 	writer.add_image(name+'_maps', img, n_iter)

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



def train(dRaw, dExpert, train_list, val_list, batch_size, epochs, npoints, nc, colorspace='srgb', apply_to='rgb', \
		  downsample_strategy='avgpool', deltae=96, lr=0.001, weight_decay=0.0, dSemSeg='', dSaliency='', nclasses=150, \
		  exp_name='', weights_from=''):
		# define summary writer
		expname = '{}_{}_np_{:d}_nf_{:d}_lr_{:.6f}_wd_{:.6f}_{}'.format(apply_to,colorspace,npoints,nc,lr,weight_decay,downsample_strategy)
		if os.path.isdir(dSemSeg): expname += '_sem'
		if os.path.isdir(dSaliency): expname += '_sal'
		if exp_name: expname += '_{}'.format(exp_name)
		writer = SummaryWriter(os.path.join('./logs/', time.strftime('%Y-%m-%d %H:%M:%S'), expname))
		# create models dir
		if not os.path.isdir('./models/'): os.makedirs('./models/')
		# define number of experts
		if isinstance(dExpert,str): dExpert = [dExpert]
		nexperts = len(dExpert)
		# get experts names
		experts_names = []
		for i in range(len(dExpert)):
			experts_names.append([s for s in dExpert[i].split(os.sep) if s][-1])
		# define transform
		trans = customTransforms.Compose([
					# customTransforms.RandomResizedCrop(size=256, scale=(1,1.2),ratio=(0.9,1.1)), \
					# customTransforms.RandomHorizontalFlip(), \
					customTransforms.ToTensor(), \
				])
		# create dataloader
		train_data_loader = data.DataLoader(
				Dataset(dRaw, dExpert, train_list, dSemSeg, dSaliency, nclasses=nclasses, trans=trans, include_filenames=False),
				batch_size = batch_size,
				shuffle = True,
				num_workers = cpu_count(),
				# num_workers = 0,
				drop_last = False
		)
		# create neural spline
		nch = 3
		if os.path.isdir(dSemSeg): nch += nclasses
		if os.path.isdir(dSaliency): nch += 1
		spline = NeuralSpline(npoints,nc,nexperts,colorspace=colorspace,apply_to=apply_to,downsample_strategy=downsample_strategy,n_input_channels=nch).cuda()
		# define optimizer
		optimizer = torch.optim.Adam(spline.parameters(), lr=lr, weight_decay=weight_decay)
		# ToDo: load weigths
		start_epoch = 0
		if weights_from:
			state = torch.load(weights_from)
			spline.load_state_dict(state['state_dict'])
			optimizer.load_state_dict(state['optimizer'])
			start_epoch = state['nepoch']
		# for each batch
		curr_iter,best_de = 0,0
		for nepoch in range(start_epoch, epochs):
			for bn, images in enumerate(train_data_loader):
				spline.train()
				raw = images[0]
				experts = images[1:]
				#print(bn)
				start_time = time.time()
				# reset gradients
				spline.zero_grad()
				optimizer.zero_grad()
				# send to GPU
				raw = raw.cuda()
				for i in range(len(experts)): experts[i] = experts[i].cuda()
				# force gradient saving
				raw.requires_grad = True
				# apply spline transform
				out, splines = spline(raw)
				# convert to lab
				out_lab, gt_lab = [],[]
				for i in range(len(experts)): gt_lab.append(spline.rgb2lab(experts[i]))
				if apply_to=='rgb':
					for i in range(len(out)): out_lab.append(spline.rgb2lab(out[i]))
				else:
					out_lab = out
				# calculate loss
				losses, loss = [], 0
				for i in range(len(out_lab)):
					if deltae == 94:
						cur_loss = ptcolor.deltaE94(out_lab[i], gt_lab[i])
					else:
						cur_loss = ptcolor.deltaE(out_lab[i], gt_lab[i])
					cur_loss = cur_loss.mean()
					losses.append(cur_loss)
					writer.add_scalar('train_loss_{}'.format(experts_names[i]), cur_loss.data.cpu().mean(), curr_iter)
					loss += cur_loss
				# divide loss by the number of experts
				loss /= len(out_lab)
				# add scalars
				writer.add_scalar('train_loss', loss.cpu().mean(), curr_iter)
				# backprop
				loss.backward()
				# update weights
				optimizer.step()
				# update optimizer
				if bn % (100 if curr_iter < 200 else 200) == 0:
					showImage(writer, raw, 'train_input', curr_iter)
					for i in range(len(experts)):
						cur_out = out[i] if apply_to=='rgb' else spline.lab2rgb(out[i])
						showImage(writer, cur_out.detach(), 'train_output_'+experts_names[i], curr_iter)
						showImage(writer, experts[i], 'train_gt_'+experts_names[i], curr_iter)
						plotSplines(writer, splines[i], 'splines_'+experts_names[i], curr_iter)
					# add histograms
					for name, param in spline.named_parameters():
						try:
							writer.add_histogram(name, param.detach().cpu().numpy(), curr_iter)
						except:
							pass
				if bn % 100 == 0:
					torch.save({
						'state_dict': spline.state_dict(),
						'optimizer': optimizer.state_dict(),
						'nepoch' : nepoch,
					}, './models/{}.pth'.format(expname))
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
					).format(nepoch,bn,train_data_loader.__len__(),curr_iter, elapsed_time, loss.item())
				print(s)
				# update iter num
				curr_iter = curr_iter + 1
			# at the end of each epoch, test values
			de, l1_l = test(dRaw, dExpert, val_list, batch_size, spline, deltae=deltae, dSemSeg=dSemSeg, \
															dSaliency=dSaliency, nclasses=150, outdir='')
			for i in range(len(de)):
				cur_exp_name = experts_names[i]
				writer.add_scalar('dE{:d}_'.format(deltae)+cur_exp_name, de[i], curr_iter)
				writer.add_scalar('L1-L_'+cur_exp_name, l1_l[i], curr_iter)
			# save best model
			testid = 2 if len(experts_names)>=4 else 0
			if nepoch == 0 or (nepoch>0 and de[testid]<best_de):
				best_de = de[testid]
				torch.save({
					'state_dict': spline.state_dict(),
					'optimizer': optimizer.state_dict(),
					'nepoch' : nepoch,
					'dE{:d}'.format(deltae) : de[testid],
				}, './models/{}_best.pth'.format(expname))
			# print
			print('{}CURR:{} dE{:d} = {}{:.4f}{} - L1_L = {}{:.4f}{}'.format(cols.BLUE,cols.LIGHT_GRAY, deltae, cols.GREEN, de[testid], cols.LIGHT_GRAY, cols.GREEN, l1_l[testid], cols.ENDC))
			print('{}BEST:{} dE{:d} = {}{:.4f}{}'.format(cols.BLUE, cols.LIGHT_GRAY, deltae, cols.GREEN, best_de, cols.ENDC))


if __name__ == '__main__':
	# parse args
	parser = argparse.ArgumentParser(description='Neural Spline.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# data parameters
	parser.add_argument("-i", "--input_dir", help="The input dir containing the raw images.",
								   default="/media/flavio/Volume/datasets/fivek/raw/")
	parser.add_argument("-e", "--experts_dir", help="The experts dirs containing the gt. Can be more then one.",
						nargs='+', default=["/media/flavio/Volume/datasets/fivek/ExpertC/"])
	parser.add_argument("-tl", "--train_list", help="Training list.",
								   default="/media/flavio/Volume/datasets/fivek/train_mit.txt")
	parser.add_argument("-vl", "--val_list", help="Validation list.",
								   default="/media/flavio/Volume/datasets/fivek/test_mit_random250.txt")
	# spline params
	parser.add_argument("-np", "--npoints",   help="Number of points of the spline.",      type=int, default=10)
	parser.add_argument("-nf", "--nfilters",  help="Number of filters.",                   type=int, default=32)
	parser.add_argument("-ds", "--downsample_strategy",  help="Type of downsampling.",     type=str, default='avgpool', choices=set(('maxpool','avgpool','convs','proj')))
	# hyper-params
	parser.add_argument("-bs", "--batchsize", help="Batchsize.",                           type=int, default=60)
	parser.add_argument("-ne", "--nepochs",   help="Number of epochs. 0 avoids training.", type=int, default=2000)
	parser.add_argument("-lr", "--lr",            help="Learning rate.",                   type=float, default=0.0001)
	parser.add_argument("-wd", "--weight_decay",  help="Weight decay.",                    type=float, default=0)
	# colorspace management
	parser.add_argument("-cs", "--colorspace",  help="Colorspace to which belong images.", type=str, default='srgb', choices=set(('srgb','prophoto')))
	parser.add_argument("-at", "--apply_to",    help="Apply spline to rgb or lab images.", type=str, default='rgb', choices=set(('rgb','lab')))
	# evaluation metric
	parser.add_argument("-de", "--deltae",  help="Version of the deltaE [76, 94].",        type=int, default=94, choices=set((76,94)))
	# semantic segmentation params
	parser.add_argument("-sem", "--semseg_dir", help="Folder containing semantic segmentation. \
												If empty, model does not use semantic segmentation", default="")
	parser.add_argument("-nc", "--nclasses",  help="Number of classes of sem. seg.",       type=int, default=150)
	# saliency parameters
	parser.add_argument("-sal", "--saliency_dir", help="Folder containing semantic segmentation. \
												If empty, model does not use semantic segmentation", default="")
	# experiment name
	parser.add_argument("-en", "--expname", help="Experiment name.", default='')
	# parse arguments
	args = parser.parse_args()
	# start training
	train(args.input_dir, args.experts_dir, args.train_list, args.val_list, args.batchsize, \
		args.nepochs, args.npoints, args.nfilters, colorspace=args.colorspace, apply_to=args.apply_to, \
		downsample_strategy=args.downsample_strategy, deltae=args.deltae, lr=args.lr, weight_decay=args.weight_decay, \
		dSemSeg=args.semseg_dir, dSaliency=args.saliency_dir, nclasses=args.nclasses, exp_name=args.expname) #, weights_from=weights_from)
