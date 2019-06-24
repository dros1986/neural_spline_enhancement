import os,sys,math,time,io,argparse
from tqdm import tqdm
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


def regen(dRaw, test_list, batch_size, spline, outdir, dSemSeg='', dSaliency='', \
		nclasses=150, outdir_splines=''):
		spline.eval()
		# create folder
		if outdir and not os.path.isdir(outdir): os.makedirs(outdir)
		if outdir_splines and not os.path.isdir(outdir_splines): os.makedirs(outdir_splines)
		# create dataloader
		test_data_loader = data.DataLoader(
				Dataset(dRaw, None, test_list, dSemSeg, dSaliency, nclasses=nclasses, include_filenames=True),
				batch_size = batch_size,
				shuffle = True,
				num_workers = cpu_count(),
				# num_workers = 0,
				drop_last = False
		)
		# regenerate all images
		for bn, (images,fns) in enumerate(tqdm(test_data_loader)):
			# get current raw images
			raw = images[0]
			# to GPU
			raw = raw.cuda()
			# compute transform
			out, splines = spline(raw)
			# detach all
			out = [e.detach() for e in out]
			# for each expert
			for i in range(len(out)):
				# define its output folders
				cur_exp_img_dir = os.path.join(outdir, str(i))
				cur_exp_spl_dir = os.path.join(outdir_splines, str(i))
				# create her/his output images
				os.makedirs(cur_exp_img_dir, exist_ok=True)
				os.makedirs(cur_exp_spl_dir, exist_ok=True)
				# get size of images
				h,w = out[i].size(2),out[i].size(3)
				# convert gt and output in lab (remember that spline in test/lab converts back in rgb)
				ot_lab = spline.rgb2lab(out[i].cuda())
				# save each image
				for j in range(out[i].size(0)):
					cur_fn = fns[j]
					cur_img = out[i][j,:,:,:].cpu().numpy().transpose((1,2,0))
					cur_img = (cur_img*255).astype(np.uint8)
					cur_img = Image.fromarray(cur_img)
					cur_img.save(os.path.join(cur_exp_img_dir, cur_fn))
				# do splines if required
				if outdir_splines:
					for j in range(out[i].size(0)):
						cur_fn = fns[j]
						drawSpline(splines[i][j,:,:], my_dpi=100).save(os.path.join(cur_exp_spl_dir, cur_fn))


if __name__ == '__main__':
	# parse args
	parser = argparse.ArgumentParser(description='Neural Spline.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# data parameters
	parser.add_argument("-i", "--input_dir", help="The input dir containing the raw images.",
								   default="/home/flavio/datasets/fivek_siggraph2018_mit/raw")
	parser.add_argument("-l", "--test_list", help="File containing filenames.",
								   default="/home/flavio/datasets/fivek_siggraph2018_mit/test-list.txt")
	# spline params
	parser.add_argument("-md", "--model", help="pth file containing the state dict of the model.", default="")
	parser.add_argument("-np", "--npoints",   help="Number of points of the spline.", type=int, default=10)
	parser.add_argument("-nf", "--nfilters",  help="Number of filters.", type=int, default=8)
	parser.add_argument("-ne", "--nexperts",  help="Number of experts.", type=int, default=1)
	parser.add_argument("-ds", "--downsample_strategy",  help="Type of downsampling.",     type=str, default='avgpool', choices=set(('maxpool','avgpool','convs','proj')))
	# hyper-params
	parser.add_argument("-bs", "--batchsize", help="Batchsize.", type=int, default=60)
	# colorspace management
	parser.add_argument("-cs", "--colorspace",  help="Colorspace to which belong images.", type=str, default='srgb', choices=set(('srgb','prophoto')))
	parser.add_argument("-at", "--apply_to",    help="Apply spline to rgb or lab images.", type=str, default='rgb', choices=set(('rgb','lab')))
	parser.add_argument("-abs","--abs",  		help="Applies absolute value to out rgb.", action='store_true')
	# semantic segmentation params
	parser.add_argument("-sem", "--semseg_dir", help="Folder containing semantic segmentation. \
												If empty, model does not use semantic segmentation", default="")
	parser.add_argument("-nc", "--nclasses",  help="Number of classes of sem. seg.",       type=int, default=150)
	# saliency parameters
	parser.add_argument("-sal", "--saliency_dir", help="Folder containing semantic segmentation. \
												If empty, model does not use semantic segmentation", default="")
	# outdir
	parser.add_argument("-od", "--out_dir",         help="Output directory.", default="")
	parser.add_argument("-ods","--out_dir_splines", help="Output directory for splines.",default="")
	# parse arguments
	args = parser.parse_args()
	# create output folder
	if not os.path.join(args.out_dir): os.makedirs(args.out_dir)
	# create net
	nch = 3
	if os.path.isdir(args.semseg_dir): nch += args.nclasses
	if os.path.isdir(args.saliency_dir): nch += 1
	spline = NeuralSpline(args.npoints,args.nfilters,args.nexperts, colorspace=args.colorspace, \
						  apply_to=args.apply_to, abs=args.abs, \
						  downsample_strategy=args.downsample_strategy, n_input_channels=nch).cuda()
	# load weights from net
	state = torch.load(args.model)
	spline.load_state_dict(state['state_dict'])

	# calculate
	regen(args.input_dir, args.test_list, args.batchsize, spline, outdir=args.out_dir, \
			dSemSeg=args.semseg_dir, dSaliency=args.saliency_dir, nclasses=args.nclasses, \
			outdir_splines=args.out_dir_splines)
