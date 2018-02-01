import os, sys, torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.utils.data as data


class Dataset(data.Dataset):
	def __init__(self, dRaw, dExpert, listfn, trans='', include_filenames=False):
			self.dRaw = dRaw
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
		# open raw and expert image
		raw = Image.open(os.path.join(self.dRaw,fn)).convert('RGB')
		exp = Image.open(os.path.join(self.dExpert,fn)).convert('RGB')
		if self.trans:
			raw,exp = self.trans([raw,exp])
		else:
			raw = transforms.ToTensor()(raw)
			exp = transforms.ToTensor()(exp)
		# return
		if self.include_filenames:
			return raw, exp, fn
		return raw, exp

	def __len__(self):
		return len(self.fns)
