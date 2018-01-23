import os,sys
from train import train
from test import test
import torch
from NeuralSpline import NeuralSpline


base_dir = '/media/flavio/Volume/datasets/fivek/'
dRaw = os.path.join(base_dir,'raw')
dExpert = os.path.join(base_dir,'ExpertC')
train_list = os.path.join(base_dir,'train_mit.txt')
val_list = os.path.join(base_dir,'test_mit_highvar50.txt')
test_list = os.path.join(base_dir,'test_mit_random250.txt')
batch_size = 70
epochs = 2000
npoints = 5 #4
nc = 32 #200
#weights_from = './checkpoints/checkpoint.pth'
weights_from = ''
train(dRaw, dExpert, train_list, val_list, batch_size, epochs, npoints, nc, weights_from=weights_from)

weights_file = './checkpoint.pth'
out_dir = './regen/'
batch_size = 10 #50
# create net
spline = NeuralSpline(npoints,nc).cuda()
# load weights from net
state = torch.load(weights_file)
spline.load_state_dict(state['state_dict'])
# calculate
l2_lab, l2_l = test(dRaw, dExpert, test_list, batch_size, spline, outdir=out_dir)
print('L2LAB = {:.4f} - L2L = {:.4f}'.format(l2_lab,l2_l))
# test(dRaw, dExpert, test_list, batch_size, npoints, nc, weights_file, out_dir)
