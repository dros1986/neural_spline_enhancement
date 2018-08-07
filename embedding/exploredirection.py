#!/usr/bin/env python3

import torch
import numpy as np
import cv2
import os
import sys
sys.path.append("..")
from NeuralSpline5 import NeuralSpline


MODEL = "../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_abcde_emb_best.pth"

MEAN = np.array([0.03112502, 0.00471976])
DIR = np.array([[-0.68501689, -0.72852718], [-0.72852718, 0.68501689]])
VARIANCE = [0.09242756571676472, 0.05093178168644688]
STEPS = np.linspace(-20, 20, 11)


def components(model, devs, n):
    proj = model.wl.weight.detach().cpu().numpy()
    mean = proj.mean(axis=1, keepdims=True)
    proj_centered = proj - mean
    evals, evecs = np.linalg.eig(np.cov(proj))
    idx = np.argsort(-evals)
    print("Experts")
    print(np.dot(proj_centered.T, evecs))
    print("Experts norm")
    print(np.dot(proj_centered.T, evecs) / np.sqrt(evals))
    steps = np.linspace(-devs, devs, 2 * n + 1)
    who = [(mean +  np.sqrt(evals[i]) *np.outer(evecs[:, i], steps)).T
           for i in idx]
    return who
    

def process(model, who, dir, filename, outdir):
    im = cv2.imread(filename)
    im = im[:, :, ::-1].astype(np.float32) / 255.0
    im = np.transpose(im, [2, 0, 1])
    im = torch.tensor(im[np.newaxis, ...], dtype=torch.float32)
    batch = im.repeat(who.shape[0], 1, 1, 1).cuda()
    with torch.no_grad():
        who = torch.tensor(who, dtype=torch.float32).cuda()
        outs = model(batch, who)
    outs = outs[0][0].cpu().numpy()
    
    for i in range(outs.shape[0]):
        im = (outs[i, ...] * 255.0)
        im = np.transpose(im, [1, 2, 0])
        im = im[:, :, ::-1].astype(np.uint8)
        outname = os.path.basename(filename)
        outname = os.path.splitext(outname)[0]
        outname = "%s-%d-%02d.png" % (outname, dir, i)
        cv2.imwrite(os.path.join(outdir, outname), im)


def main():
    if len(sys.argv) < 3:
        print("USAGE: /.exploredirection.py IM1 ... IMk OUTDIR")
        return
    model = NeuralSpline(10, 8, 1).cuda()
    data = torch.load(MODEL)
    model.load_state_dict(data["state_dict"])
    model.eval()
    who = components(model, 10, 10)
    outdir = sys.argv[-1]
    for imgname in sys.argv[1:-1]:
        for dir in (0, 1):
            process(model, who[dir], dir, imgname, outdir)
    

if __name__ == "__main__":
    main()
