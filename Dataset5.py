import os, sys, torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.utils.data as data


class Dataset(data.Dataset):
        def __init__(self, rawdir, expdir, listfile):
                with open(listfile) as f:
                        self.files = [x.strip() for x in f if x.strip()]
                self.rawdir = rawdir
                self.expdir = expdir[0]
                self.experts = "expA expB expC expD expE".split()

        def _load_image(self, name):
                image = Image.open(name).convert('RGB')
                a = np.array(image, np.float32)
                a = a.transpose([2, 0, 1]) / 255.0
                return a
                
        def __getitem__(self, index):
                ifile, iexpert = divmod(index, len(self.experts))
                raw_name = os.path.join(self.rawdir, self.files[ifile])
                raw = self._load_image(raw_name)
                exp_name = os.path.join(self.expdir, self.experts[iexpert], self.files[ifile])
                target = self._load_image(exp_name)
                v = np.zeros(len(self.experts), dtype=np.float32)
                v[iexpert] = 1
                # !!! Horizontal flip?
                return torch.tensor(raw), torch.tensor(target), torch.tensor(v)
        
        def __len__(self):
                return len(self.files) * len(self.experts)


def _test():
        dataset = Dataset("/mnt/data/dataset/fivek/siggraph2018/256x256/raw",
                          ["/mnt/data/dataset/fivek/siggraph2018/256x256"],
                          "/mnt/data/dataset/fivek/siggraph2018/test-list.txt")
                          
        loader = data.DataLoader(dataset, batch_size=10, shuffle=True)
        for r, e, v in loader:
                print("x".join(map(str, r.size())), r.min().item(), r.max().item(), "x".join(map(str, e.size())), e.min().item(), e.max().item(), "x".join(map(str, v.size())))


if __name__ == "__main__":
        _test()
