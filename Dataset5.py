import os, sys, torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.utils.data as data


ENCODING = dict((x, n) for (n, x) in enumerate("expA expB expC expD expE".split()))
ENCODING.update(dict((x, 2) for x in "user-F user-G user-H user-I user-J".split()))
ENCODING[""] = -1


class Dataset(data.Dataset):
        def __init__(self, rawdir, expdir, listfile, experts, istrain):
                with open(listfile) as f:
                        self.files = [x.strip() for x in f if x.strip()]
                self.rawdir = rawdir
                self.expdir = expdir[0]
                # experts: sequence of (index, directory) pairs
                self.experts = list(experts)
                self.istrain = istrain

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
                bit = ENCODING[self.experts[iexpert]]
                who = torch.tensor([i == bit for i in range(len(ENCODING))], dtype=torch.float32)
                # Horizontal flip
                if self.istrain and np.random.random() < 0.5:
                        raw = raw[:, :, ::-1]
                        target = target[:, :, ::-1]
                # Random crop
                # csz = 32
                # if self.istrain:
                #         i = np.random.randint(csz)
                #         j = np.random.randint(csz)
                # else:
                #         i = j = 16
                # _, h, w = raw.shape
                # raw = raw[:, i:(h - csz + i), j:(w - csz + j)]
                # target = target[:, i:(h - csz + i), j:(w - csz + j)]
                raw_t = torch.from_numpy(raw.copy())
                target_t = torch.from_numpy(target.copy())
                return raw_t, target_t, who

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
