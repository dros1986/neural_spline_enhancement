import os, sys, torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.utils.data as data


class Dataset(data.Dataset):
	def __init__(self, dRaw, dExpert, listfn, dSemSeg='', dSaliency='', nclasses=150, trans='', include_filenames=False):
			self.dRaw = dRaw
			if isinstance(dExpert, str):
				dExpert = [dExpert]
			self.dExpert = dExpert
			self.dSemSeg = dSemSeg
			self.dSaliency = dSaliency
			self.include_filenames = include_filenames
			self.listfn = listfn
			self.trans = trans
			self.nclasses = nclasses
			# read file with filenames
			in_file = open(listfn,"r")
			text = in_file.read()
			in_file.close()
			# get filenames
			self.fns = [l for l in text.split('\n') if l]

	def __getitem__(self, index):
		fn = self.fns[index]
		# open images
		raw = np.array(Image.open(os.path.join(self.dRaw,fn)).convert('RGB'))
		raw = raw.astype(np.float32) / 255.
		images =[raw]
		# check if there are experts to load
		if self.dExpert is not None:
			# if there are, load their images
			for cur_dexp in self.dExpert:
				cur_img = np.array(Image.open(os.path.join(cur_dexp,fn)).convert('RGB'))
				cur_img = cur_img.astype(np.float32) / 255.
				images.append(cur_img)
		# open semantic segmentation
		semseg = []
		if os.path.isdir(self.dSemSeg):
			semseg_img = np.array(Image.open(os.path.join(self.dSemSeg,fn)))
			chs = []
			for i in range(self.nclasses):
				cur_ch = np.zeros((images[0].shape[0],images[0].shape[1]))
				cur_ch[semseg_img==i]=1
				chs.append(np.expand_dims(cur_ch,axis=2))
			semseg_maps = np.concatenate(chs,axis=2).astype(np.float32)
			images[0] = np.concatenate((images[0], semseg_maps),axis=2)
		# open saliency
		if os.path.isdir(self.dSaliency):
			saliency_img = np.array(Image.open(os.path.join(self.dSaliency,fn)))
			saliency_img = saliency_img.astype(np.float) / 255.
			saliency_img = np.expand_dims(saliency_img,axis=2).astype(np.float32)
			images[0] = np.concatenate((images[0], saliency_img),axis=2)
		# apply transforms
		if self.trans:
			images = self.trans(images)
		else:
			images = [transforms.ToTensor()(images[i]) for i in range(len(images))]
		# return
		if self.include_filenames:
			return images, fn
		return images # first the raw, then the experts

	def __len__(self):
		return len(self.fns)
