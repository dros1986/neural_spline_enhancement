import os, sys, torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.utils.data as data


class Dataset(data.Dataset):
	def __init__(self, dRaw, dExpert, listfn, trans='', include_filenames=False):
			self.dRaw = dRaw
			if isinstance(dExpert, str):
				dExpert = [dExpert]
			self.dExpert = dExpert
			self.include_filenames = include_filenames
			self.listfn = listfn
			self.trans = trans
			# read file with filenames
			in_file = open(listfn,"r")
			text = in_file.read()
			in_file.close()
			# get filenames
			self.fns = [l for l in text.split('\n') if l]

	def __getitem__(self, index):
		fn = self.fns[index]
		# open images
		images = [Image.open(os.path.join(self.dRaw,fn)).convert('RGB')]
		for cur_dexp in self.dExpert:
			images.append(Image.open(os.path.join(cur_dexp,fn)).convert('RGB'))
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
