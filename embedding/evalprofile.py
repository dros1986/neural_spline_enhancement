import torch
import numpy as np
import sys
sys.path.append("..")
from Dataset5 import Dataset
from NeuralSpline5 import NeuralSpline
import ptcolor


PROFILE = [0.0, 0.0, 0.0, 0.0, 1.0]
MODEL = "../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_abcde_emb_best.pth"
TARGET_EXP = "expE"


RAWDIR = "/mnt/data/dataset/fivek/siggraph2018/256x256/raw"
EXPDIR = "/mnt/data/dataset/fivek/siggraph2018/256x256"
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


def eval_profile(profile, model, raw_dir, expert_dir, file_list, expert, verbose=False):
    loader = torch.utils.data.DataLoader(
	Dataset(raw_dir, [expert_dir], file_list, ["exp" + expert.upper()], False),
	batch_size=16,
	shuffle=False,
	num_workers=torch.get_num_threads(),
	drop_last=False)

    device = next(model.parameters()).device
    who = torch.tensor(profile).to(device)
    nimages = 0
    tot_de76 = 0
    tot_de94 = 0
    tot_dl = 0
    for images in loader:
        raw = images[0].to(device)
        expert = images[1].to(device)
        bs = expert.size(0)
        nimages += bs
        with torch.no_grad():
            out = model(raw, who.repeat(bs, 1))[0][0]
            de76, de94, dl = metrics(model, out, expert)
            tot_de76 += torch.sum(de76).item()
            tot_de94 += torch.sum(de94).item()
            tot_dl += torch.sum(dl).item()
        if verbose:
            print(".", flush=True, end="")
    return (tot_de76 / nimages,  tot_de94 / nimages, tot_dl / nimages)


def main():
    model = NeuralSpline(10, 8, 1)
    model_file = MODEL
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    target_expert = "c"
    if len(sys.argv) > 2:
        target_expert = sys.argv[2]
    profile = PROFILE
    if len(sys.argv) > 3:
        profile = list(map(float, sys.argv[3:8]))
        
    data = torch.load(model_file)
    model.load_state_dict(data["state_dict"])
    model.cuda()
    model.eval()

    de76, de94, dl = eval_profile(profile, model, RAWDIR, EXPDIR, TEST_LIST, target_expert, verbose=True)
    print()
    print("DE76", de76)
    print("DE94", de94)
    print("DL  ", dl)
    

if __name__ == "__main__":
    main()
