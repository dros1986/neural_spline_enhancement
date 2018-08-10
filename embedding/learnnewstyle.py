import torch
import numpy as np
import random
import sys
sys.path.append("..")
from Dataset5 import Dataset
from NeuralSpline5 import NeuralSpline
import ptcolor


MODEL = "../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_abcde_emb_best.pth"
RAWDIR = "/mnt/data/dataset/fivek/siggraph2018/256x256/raw"


def _load_images(raw_dir, style_dir, list_file):
    dataset = Dataset(raw_dir, [style_dir], list_file, [""], False)
    with open(list_file) as f:
        nimages = len([x for x in f if x.strip()])
    raw = torch.zeros(nimages, 3, 256, 256)
    target = torch.zeros(nimages, 3, 256, 256)
    n = len(dataset)
    for i, index in enumerate(random.sample(range(n), nimages)):
        r, t, _ = dataset[index]
        raw[i, ...] = r
        target[i, ...] = t
    return raw, target


def fit_profile(model, list_file, raw_dir, style_dir,
                iterations=100, batch_size=10, learning_rate=0.1,
                verbose=False):
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    mask = torch.ones(5).to(device)
    who = (torch.ones(5) / 5.0).to(device)
    who.requires_grad = True
    raw_cpu, target_cpu = _load_images(raw_dir, style_dir, list_file)
    if verbose:
        print("Load", raw_cpu.size(0), "images")
    nimages = raw_cpu.size(0)
    target_cpu = model.rgb2lab(target_cpu)
    if verbose:
        print("Target images converted to LAB")
    optimizer = torch.optim.Adam([who], lr=learning_rate, weight_decay=0)
    for step in range(1, iterations + 1):
        b = step % ((nimages + batch_size - 1) // batch_size)
        raw = raw_cpu[b * batch_size:(b + 1) * batch_size, ...].to(device)
        target = target_cpu[b * batch_size:(b + 1) * batch_size, ...].to(device)
        optimizer.zero_grad()
        out, splines = model(raw, (who * mask).repeat(raw.size(0), 1))
        out_lab = model.rgb2lab(out[0])
        loss = ptcolor.deltaE(out_lab, target).mean()
        loss.backward()
        optimizer.step()
        if verbose:
            print("{:d}  {:.3f}   {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(step, loss.item(), *who.detach().cpu().numpy()))
    return who.detach().cpu().numpy()


def main():
    model = NeuralSpline(10, 8, 1)
    data = torch.load(MODEL)
    model.load_state_dict(data["state_dict"])
    model.cuda()
    list_file = "gs.txt"   # !!!
    style_dir = "./gs"
    profile = fit_profile(model, list_file, RAWDIR, style_dir, verbose=True)
    print()
    print(*profile)


if __name__ == "__main__":
    main()
