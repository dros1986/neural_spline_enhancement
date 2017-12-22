import os,sys
from train import train



base_dir = '/media/flavio/Volume/datasets/fivek/'
dRaw = os.path.join(base_dir,'raw')
dExpert = os.path.join(base_dir,'ExpertC')
train_list = os.path.join(base_dir,'train_mit.txt')
val_list = os.path.join(base_dir,'test_mit_highvar50.txt')
batch_size = 50
epochs = 2000
npoints = 4#2 #10
#weights_from = './checkpoints/checkpoint.pth'
weights_from = ''
train(dRaw, dExpert, train_list, val_list, batch_size, epochs, npoints, weights_from=weights_from)
