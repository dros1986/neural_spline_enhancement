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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import ptcolor
from tqdm import tqdm


def rgb2lab(x, colorspace):
	if colorspace=='srgb':
		white_point = 'd65'
		gamma_correction = 'srgb'
	else:
		white_point = 'd50'
		gamma_correction = 1.8
	return ptcolor.rgb2lab(x,	white_point=white_point,           \
								gamma_correction=gamma_correction, \
								clip_rgb=True,						\
								space=colorspace)

def lab2rgb(x,colorspace):
	if colorspace=='srgb':
		white_point = 'd65'
		gamma_correction = 'srgb'
	else:
		white_point = 'd50'
		gamma_correction = 1.8
	return ptcolor.lab2rgb(x, 	white_point=white_point,           \
								gamma_correction=gamma_correction, \
								clip_rgb=True,				\
								space=colorspace)



def test_spline(dRaw, dExpert, test_list, batch_size, spline, deltae=94, dSemSeg='', dSaliency='', \
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
				Dataset(dRaw, dExpert, test_list, dSemSeg, dSaliency, nclasses=nclasses, include_filenames=True),
				batch_size = batch_size,
				shuffle = True,
				num_workers = cpu_count(),
				# num_workers = 0,
				drop_last = False
		)
		# create function for calculating L1
		def L1(gt_lab, ot_lab):
			return torch.abs(ot_lab[:,0,:,:]-gt_lab[:,0,:,:])
		# create outputs
		acc = [[0,0,0] for i in range(len(dExpert))]
		fun = [ptcolor.deltaE, ptcolor.deltaE94, L1]
		nme = ['dE76','dE94','L1']
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
				# calculate metrics
				for nn in range(len(nme)):
					acc[i][nn] += fun[nn](gt_lab, ot_lab).sum()
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
		# normalize
		for ne in range(len(out)):
			for nm in range(len(nme)):
				acc[ne][nm] /= nimages*h*w
		# return
		return nme, acc



def test_images(dRes, dExpert, test_list, batch_size, colorspace='srgb', dSemSeg='', dSaliency='', \
		nclasses=150, outdir=''):
		# get expert name and create corresponding folder
		expert_name = dExpert.split(os.sep)[-1]
		# create dataloader
		test_data_loader = data.DataLoader(
				Dataset(dRes, dExpert, test_list, dSemSeg, dSaliency, nclasses=nclasses, include_filenames=True),
				batch_size = batch_size,
				shuffle = True,
				num_workers = cpu_count(),
				# num_workers = 0,
				drop_last = False
		)
		# create function for calculating L1
		def L1(gt_lab, ot_lab):
			return torch.abs(ot_lab[:,0,:,:]-gt_lab[:,0,:,:])
		# create outputs
		acc = [0,0,0]
		fun = [ptcolor.deltaE, ptcolor.deltaE94, L1]
		nme = ['dE76','dE94','L1']
		nimages = 0
		# calculate differences
		for bn, (images,fns) in enumerate(tqdm(test_data_loader)):
			out = images[0]
			expert = images[1]
			nimages += expert.size(0)
			# to GPU
			out = out.cuda()
			expert = expert.cuda()
			# get size of images
			h,w = out.size(2),out.size(3)
			# convert gt and output in lab
			ot_lab = rgb2lab(out,colorspace=colorspace)
			gt_lab = rgb2lab(expert,colorspace=colorspace)
			# calculate metrics
			for i in range(len(nme)):
				acc[i] += fun[i](gt_lab, ot_lab).sum()
		# normalize
		for i in range(len(nme)):
			acc[i] /= nimages*h*w
		# return
		return nme, acc


if __name__ == '__main__':
	# parse args
	parser = argparse.ArgumentParser(description='Neural Spline.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# data parameters
	parser.add_argument("-i", "--input_dir", help="The input dir containing the res images.",
								   default="/media/flavio/Volume/datasets/fivek/res/")
	parser.add_argument("-e", "--expert_dir", help="The expert dir containing the gt.",
								   default="/media/flavio/Volume/datasets/fivek/ExpertC/")
	parser.add_argument("-l", "--test_list", help="File containing filenames.",
								   default="/media/flavio/Volume/datasets/fivek/test_mit_random250.txt")
	# hyper-params
	parser.add_argument("-bs", "--batchsize", help="Batchsize.",                           type=int, default=60)
	# colorspace management
	parser.add_argument("-cs", "--colorspace",  help="Colorspace to which belong images.", type=str, default='srgb', choices=set(('srgb','prophoto')))
	parser.add_argument("-at", "--apply_to",    help="Apply spline to rgb or lab images.", type=str, default='rgb', choices=set(('rgb','lab')))
	parser.add_argument("-abs","--abs",  		help="Applies absolute value to out rgb.", action='store_true')
	# evaluation metric
	# parser.add_argument("-de", "--deltae",  help="Version of the deltaE [76, 94].",        type=int, default=94, choices=set((76,94)))
	# semantic segmentation params
	parser.add_argument("-sem", "--semseg_dir", help="Folder containing semantic segmentation. \
												If empty, model does not use semantic segmentation", default="")
	parser.add_argument("-nc", "--nclasses",  help="Number of classes of sem. seg.",       type=int, default=150)
	# saliency parameters
	parser.add_argument("-sal", "--saliency_dir", help="Folder containing semantic segmentation. \
												If empty, model does not use semantic segmentation", default="")
	# outdir
	parser.add_argument("-od", "--out_dir", help="Output directory.", default="")
	# parse arguments
	args = parser.parse_args()
	# calculate
	nme,acc = test_images(args.input_dir, args.expert_dir, args.test_list, args.batchsize, \
					colorspace=args.colorspace, dSemSeg=args.semseg_dir, \
					dSaliency=args.saliency_dir, nclasses=args.nclasses, outdir=args.out_dir)
	# print results
	txt = ' - '.join(['{} = {:.4f}'.format(cur_name,cur_val) for cur_name,cur_val in zip(nme,acc)])
	print(txt)
