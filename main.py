import os,sys
import argparse
from train import train
from test import test
import torch
from NeuralSpline import NeuralSpline


# parse args
parser = argparse.ArgumentParser(description='Neural Spline.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", "--input_dir", help="The input dir containing the raw images.",
							   default="/media/flavio/Volume/datasets/fivek/raw/")
parser.add_argument("-e", "--experts_dir", help="The experts dirs containing the gt. Can be more then one.",
					nargs='+', default=["/media/flavio/Volume/datasets/fivek/ExpertC/"])

parser.add_argument("-bs", "--batchsize", help="Batchsize.",                           type=int, default=60)
parser.add_argument("-np", "--npoints",   help="Number of points of the spline.",      type=int, default=10)
parser.add_argument("-ne", "--nepochs",   help="Number of epochs. 0 avoids training.", type=int, default=2000)
parser.add_argument("-nf", "--nfilters",  help="Number of filters.",                   type=int, default=32)

parser.add_argument("-lr", "--lr",            help="Learning rate.",                   type=float, default=0.001)
parser.add_argument("-wd", "--weight_decay",  help="Weight decay.",                    type=float, default=0)

parser.add_argument("-cs", "--colorspace",  help="Colorspace to which is applied the spline.", type=str, default='rgb')
parser.add_argument("-de", "--deltae",  help="Version of the deltaE [76, 94].",        type=int, default=94)

parser.add_argument("-ds", "--downsample_strategy",  help="Type of downsampling. Accepted values are: \
													[maxpool, avgpool, convs]",     type=str, default='avgpool')

parser.add_argument("-tr", "--train", help="Train. Lunch with args <train_txt> <val_txt>",
					nargs='+', default=["/media/flavio/Volume/datasets/fivek/train_mit.txt", \
										"/media/flavio/Volume/datasets/fivek/test_mit_highvar50.txt", \
										""
										])
parser.add_argument("-ts", "--test", help="Test. Lunch with arg <test_txt> <model> <outdir>",
					nargs='+', default=["/media/flavio/Volume/datasets/fivek/test_mit_random250.txt"])

parser.add_argument("-en", "--expname", help="Experiment name.", default='')

args = parser.parse_args()

# check args
btrain,btest = True,True
if not (len(args.train)>=2 and args.batchsize > 0): btrain = False
if not len(args.test)==3: btest = False

# train if required
if btrain:
	train(args.input_dir, args.experts_dir, args.train[0], args.train[1], args.batchsize, args.nepochs, \
		args.npoints, args.nfilters, apply_to=args.colorspace, downsample_strategy=args.downsample_strategy, \
		deltae=args.deltae, lr=args.lr, weight_decay=args.weight_decay, exp_name=args.expname) #, weights_from=weights_from)

# test
if btest:
	# create net
	spline = NeuralSpline(args.npoints,args.nfilters,len(args.experts_dir),apply_to=args.colorspace,downsample_strategy=args.downsample_strategy).cuda()
	# load weights from net
	state = torch.load(args.test[1])
	spline.load_state_dict(state['state_dict'])
	# calculate
	l2_lab, l2_l = test(args.input_dir, args.experts_dir, args.test[0], args.batchsize, spline, outdir=args.test[2])
	for i in range(len(l2_lab)):
		print('{:d}: L2LAB = {:.4f} - L2L = {:.4f}'.format(i,l2_lab[i],l2_l[i]))
