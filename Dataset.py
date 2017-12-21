import os, sys, torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.utils.data as data


class Dataset(data.Dataset):
	def __init__(self, dRaw, dExpert, listfn, include_filenames=False):
			self.dRaw = dRaw
			self.dExpert = dExpert
			self.include_filenames = include_filenames
			self.listfn = listfn
			# read file with filenames
			in_file = open(listfn,"r")
			text = in_file.read()
			in_file.close()
			# get filenames
			self.fns = [l for l in text.split('\n') if l]

	def __getitem__(self, index):
		fn = self.fns[index]
		# open raw and expert image
		raw = transforms.ToTensor()(Image.open(os.path.join(self.dRaw,fn)).convert('RGB'))
		exp = transforms.ToTensor()(Image.open(os.path.join(self.dExpert,fn)).convert('RGB'))
		# return
		if self.include_filenames:
			return raw, exp, fn
		return raw, exp

	def __len__(self):
		return len(self.fns)
