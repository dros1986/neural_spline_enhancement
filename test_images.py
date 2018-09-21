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


def test(dRes, dExpert, test_list, batch_size, deltae=94, colorspace='srgb', dSemSeg='', dSaliency='', \
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
		# create outputs
		de,diff_l,nimages = 0,0,0
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
			# calculate deltaE
			if deltae == 94:
				cur_de = ptcolor.deltaE94(gt_lab, ot_lab)
			else:
				cur_de = ptcolor.deltaE(gt_lab, ot_lab)
			# add current deltaE to accumulator
			de += cur_de.sum()
			# calculate L1 on L channel and add to
			diff_l += torch.abs(ot_lab[:,0,:,:]-gt_lab[:,0,:,:]).sum()
		# calculate differences
		de /= nimages*h*w
		diff_l /= nimages*h*w
		# return values
		return de, diff_l


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
	parser.add_argument("-de", "--deltae",  help="Version of the deltaE [76, 94].",        type=int, default=94, choices=set((76,94)))
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
	de, l1_l = test(args.input_dir, args.expert_dir, args.test_list, args.batchsize, \
					args.deltae, colorspace=args.colorspace, dSemSeg=args.semseg_dir, \
					dSaliency=args.saliency_dir, nclasses=args.nclasses, outdir=args.out_dir)
	# print results
	print('dE{:d} = {:.4f} - L1L = {:.4f}'.format(args.deltae,de,l1_l))
	# for i in range(len(de)):
	# 	print('{:d}: dE{:d} = {:.4f} - L1L = {:.4f}'.format(i,args.deltae,de[i],l1_l[i]))
