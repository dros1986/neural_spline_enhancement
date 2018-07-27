import os,sys,math,time,io,argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, utils
from Dataset5 import Dataset
from NeuralSpline5 import NeuralSpline
from tensorboardX import SummaryWriter
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import ptcolor


def drawSpline(cur_spline, my_dpi=100):
	# define range
	r = torch.arange(0,1,1.0/cur_spline.size(1)).numpy()
	# open figure
	plt.figure(figsize=(400/my_dpi, 400/my_dpi), dpi=my_dpi)
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
	return Image.open(buf)


def test(dRaw, dExpert, test_list, batch_size, spline, deltae=94, dSemSeg='', dSaliency='', \
		nclasses=150, outdir='', outdir_splines=''):
		spline.eval()
		# create folder
		if outdir and not os.path.isdir(outdir): os.makedirs(outdir)
		if outdir_splines and not os.path.isdir(outdir_splines): os.makedirs(outdir_splines)
		# get experts names and create corresponding folder
		experts_names = []
		for i in range(len(dExpert)):
			experts_names.append([s for s in dExpert[i].split(os.sep) if s][-1])
			if outdir and not os.path.isdir(os.path.join(outdir,experts_names[-1])):
				os.makedirs(os.path.join(outdir,experts_names[-1]))
			if outdir_splines and not os.path.isdir(os.path.join(outdir_splines,experts_names[-1])):
				os.makedirs(os.path.join(outdir_splines,experts_names[-1]))
		# create dataloader
		test_data_loader = data.DataLoader(
				Dataset(dRaw, dExpert, test_list),
				batch_size = batch_size,
				shuffle = True,
				num_workers = cpu_count(),
				# num_workers = 0,
				drop_last = False
		)
		# create output mat
		de,diff_l,nimages = [0 for i in range(len(dExpert))],[0 for i in range(len(dExpert))],0
		# calculate differences
		for bn, images in enumerate(test_data_loader):
			raw = images[0]
			experts = [images[1]]
			who = images[2]
			nimages += experts[0].size(0)
			# to GPU
			raw = raw.cuda()
			who = who.cuda()
			# compute transform
			out, splines = spline(raw, who)
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
				# do splines if required
				if outdir_splines:
					for j in range(out[i].size(0)):
						cur_fn = fns[j]
						drawSpline(splines[i][j,:,:], my_dpi=100).save(os.path.join(outdir_splines,experts_names[i],cur_fn))
		# calculate differences
		for i in range(len(de)):
			de[i] /= nimages*h*w
			diff_l[i] /= nimages*h*w
		# return values
		return de, diff_l


if __name__ == '__main__':
	# parse args
	parser = argparse.ArgumentParser(description='Neural Spline.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# data parameters
	parser.add_argument("-i", "--input_dir", help="The input dir containing the raw images.",
								   default="/media/flavio/Volume/datasets/fivek/raw/")
	parser.add_argument("-e", "--experts_dir", help="The experts dirs containing the gt. Can be more then one.",
						nargs='+', default=["/media/flavio/Volume/datasets/fivek/ExpertC/"])
	parser.add_argument("-l", "--test_list", help="File containing filenames.",
								   default="/media/flavio/Volume/datasets/fivek/test_mit_random250.txt")
	# spline params
	parser.add_argument("-md", "--model", help="pth file containing the state dict of the model.", default="")
	parser.add_argument("-np", "--npoints",   help="Number of points of the spline.",      type=int, default=10)
	parser.add_argument("-nf", "--nfilters",  help="Number of filters.",                   type=int, default=32)
	parser.add_argument("-ds", "--downsample_strategy",  help="Type of downsampling.",     type=str, default='avgpool', choices=set(('maxpool','avgpool','convs','proj')))
	# hyper-params
	parser.add_argument("-bs", "--batchsize", help="Batchsize.",                           type=int, default=60)
	# colorspace management
	parser.add_argument("-cs", "--colorspace",  help="Colorspace to which belong images.", type=str, default='srgb', choices=set(('srgb','prophoto')))
	parser.add_argument("-at", "--apply_to",    help="Apply spline to rgb or lab images.", type=str, default='rgb', choices=set(('rgb','lab')))
	parser.add_argument("-abs","--abs",  		help="Applies absolute value to out rgb.", action='store_true')
	# evaluation metric
	parser.add_argument("-de", "--deltae",  help="Version of the deltaE [76, 94].",        type=int, default=94, choices=set((76,94)))
	# semantic segmentation params
	parser.add_argument("-sem", "--semseg_dir", help="Folder containing semantic segmentation. \
												If empty, model does not use semantic segmentation", default="")
	parser.add_argument("-nc", "--nclasses",  help="Number of classes of sem. seg.",       type=int, default=150)
	# saliency parameters
	parser.add_argument("-sal", "--saliency_dir", help="Folder containing semantic segmentation. \
												If empty, model does not use semantic segmentation", default="")
	# outdir
	parser.add_argument("-od", "--out_dir",         help="Output directory.", default="")
	parser.add_argument("-ods","--out_dir_splines", help="Output directory for splines.", default="")
	# parse arguments
	args = parser.parse_args()
	# create output folder
	if not os.path.join(args.out_dir): os.makedirs(args.out_dir)
	# create net
	nch = 3
	if os.path.isdir(args.semseg_dir): nch += args.nclasses
	if os.path.isdir(args.saliency_dir): nch += 1
	spline = NeuralSpline(args.npoints,args.nfilters,len(args.experts_dir),colorspace=args.colorspace, \
						  apply_to=args.apply_to,abs=args.abs,downsample_strategy=args.downsample_strategy,  \
						  n_input_channels=nch).cuda()
	# load weights from net
	state = torch.load(args.model)
	spline.load_state_dict(state['state_dict'])
	# calculate
	de, l1_l = test(args.input_dir, args.experts_dir, args.test_list, args.batchsize, \
					spline, args.deltae, dSemSeg=args.semseg_dir, dSaliency=args.saliency_dir, \
					nclasses=args.nclasses, outdir=args.out_dir, outdir_splines=args.out_dir_splines)
	# print results for each expert
	for i in range(len(de)):
		print('{:d}: dE{:d} = {:.4f} - L1L = {:.4f}'.format(i,args.deltae,de[i],l1_l[i]))
