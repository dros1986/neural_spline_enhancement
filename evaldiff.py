#!/usr/bin/env python3


import argparse
import torch
import ptcolor
import os
import scipy.misc


def parse_args():
    parser = argparse.ArgumentParser(description='eval')
    a = parser.add_argument
    a("list")
    a("dir1")
    a("dir2")
    return parser.parse_args()


def read(d, f):
    f = os.path.join(d, f)
    im = scipy.misc.imread(f, mode="RGB")
    t = torch.from_numpy(im.transpose((2, 0, 1)) / 255.0)
    return t[None, :, :, :].to(torch.float32)


def main():
    args = parse_args()
    with open(args.list) as f:
        files = [l.strip() for l in f if l.strip()]
    tot = []
    for f in files:
        im1 = read(args.dir1, f)
        im2 = read(args.dir2, f)
        lab1 = ptcolor.rgb2lab(im1, white_point="d50", gamma_correction=1.8, space="prophoto")
        lab2 = ptcolor.rgb2lab(im2, white_point="d50", gamma_correction=1.8, space="prophoto")
        dl = torch.abs(lab1[:, 0, :, :] - lab2[:, 0, :, :])
        tot.append(torch.mean(dl).item())
        print(".", flush=True, end="")
    print()
    print(sum(tot) / len(tot))


if __name__ == "__main__":
    main()
