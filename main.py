import os,sys
from train import train
from test import test



base_dir = '/media/flavio/Volume/datasets/fivek/'
dRaw = os.path.join(base_dir,'raw')
dExpert = os.path.join(base_dir,'ExpertC')
train_list = os.path.join(base_dir,'train_mit.txt')
val_list = os.path.join(base_dir,'test_mit_highvar50.txt')
test_list = os.path.join(base_dir,'test_mit_random250.txt')
batch_size = 90 #50
epochs = 2000
npoints = 10 #4
nc = 32 #200
#weights_from = './checkpoints/checkpoint.pth'
weights_from = ''
#train(dRaw, dExpert, train_list, val_list, batch_size, epochs, npoints, nc, weights_from=weights_from)

weights_file = './checkpoint.pth'
out_dir = './regen/'
batch_size = 50
test(dRaw, dExpert, test_list, batch_size, npoints, nc, weights_file, out_dir)
