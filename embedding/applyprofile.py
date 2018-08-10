import torch
import numpy as np
import sys
sys.path.append("..")
from NeuralSpline5 import NeuralSpline
import ptcolor
import argparse
import os
from PIL import Image


MODEL = "../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_abcde_emb_best.pth"
RAWDIR = "/mnt/data/dataset/fivek/siggraph2018/256x256/raw"
TEST_LIST = "/mnt/data/dataset/fivek/siggraph2018/test-list.txt"


def metrics(model, a, b):
    laba = model.rgb2lab(a)
    labb = model.rgb2lab(b)
    d = ptcolor.deltaE(laba, labb)
    de76 = torch.mean(torch.mean(d[:, 0, :, :], 2), 1)
    d = ptcolor.deltaE94(laba, labb)
    de94 = torch.mean(torch.mean(d[:, 0, :, :], 2), 1)
    d = torch.abs(laba - labb)
    dl = torch.mean(torch.mean(d[:, 0, :, :], 2), 1)
    return de76, de94, dl


def apply_profile(profile, model, raw_dir, file_list, output_dir, verbose=False):
    device = next(model.parameters()).device
    who = torch.tensor(profile).to(device)
    os.makedirs(output_dir, exist_ok=True)

    with open(file_list) as f:
        files = [x.strip() for x in f if x.strip()]

    for f in files:
        name = f.strip()
        if not name:
            continue
        image = Image.open(os.path.join(raw_dir, name)).convert('RGB')
        a = np.array(image, np.float32)
        a = a.transpose([2, 0, 1]) / 255.0
        raw = torch.tensor(a[None, ...]).to(device)
        with torch.no_grad():
            out = model(raw, who[None, :])[0][0]
        a = out[0, ...].cpu().numpy()
        a = (a.transpose([1, 2, 0]) * 255).astype(np.uint8)
        Image.fromarray(a).save(os.path.join(output_dir, name))
        if verbose:
            print(name, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Apply a model with a profile")
    a = parser.add_argument
    a("profile", nargs=5, type=float, help="Profile of the user")
    a("--output", default="output", help="directory where output data is placed")
    a("--model", default=MODEL, help="Model file")
    return parser.parse_args()


def main():
    args = parse_args()
    model = NeuralSpline(10, 8, 1)
    model_file = args.model
    profile = args.profile
        
    data = torch.load(model_file)
    model.load_state_dict(data["state_dict"])
    model.cuda()
    model.eval()

    apply_profile(profile, model, RAWDIR, TEST_LIST, args.output, verbose=True)
    

if __name__ == "__main__":
    main()
