import torch
import numpy as np
import scipy.misc
import sys
sys.path.append("..")
from Dataset5 import Dataset
from NeuralSpline5 import NeuralSpline
import ptcolor
import os


PROFILE = [0.0, 0.0, 1.0, 0.0, 0.0]
MODEL = "../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_abcde_emb_best.pth"
TARGET_EXP = "user-F"


RAWDIR = "/mnt/data/dataset/fivek/new-users/fivek-test100-raw"
EXPDIR = "/mnt/data/dataset/fivek/new-users/"
TEST_LIST = "/mnt/data/dataset/fivek/new-users/list.txt"
OUTPUT_DIR = None  #"./out"


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
	Dataset(raw_dir, [expert_dir], file_list, ["user-" + expert.upper()], False),
	batch_size=16,
	shuffle=False,
	num_workers=torch.get_num_threads(),
	drop_last=False)

    with open(file_list) as f:
        files = [x.strip() for x in f if x.strip()][::-1]

    device = next(model.parameters()).device
    who = torch.tensor(profile).to(device).to(torch.float32)
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
            out = model(raw, who.repeat(bs, 1))[0][0].clamp(0, 1)
            de76, de94, dl = metrics(model, expert, out)
            tot_de76 += torch.sum(de76).item()
            tot_de94 += torch.sum(de94).item()
            tot_dl += torch.sum(dl).item()
        if OUTPUT_DIR is not None:
            out_images = out.cpu().numpy()
            out_images = out_images.transpose([0, 2, 3, 1]) * 255.0
            out_images = out_images.astype(np.uint8)
            for i in range(out_images.shape[0]):
                fname = os.path.join(OUTPUT_DIR, files.pop())
                scipy.misc.imsave(fname, out_images[i])
        if verbose:
            print(".", flush=True, end="")
    return (tot_de76 / nimages,  tot_de94 / nimages, tot_dl / nimages)


def main():
    # evalprofile.py  [MODEL_FILE [TARGET_EXPERT [<PROFILE>]]]
    # expert in F-J
    model = NeuralSpline(10, 8, 1)
    model_file = MODEL
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    target_expert = "f"
    if len(sys.argv) > 2:
        target_expert = sys.argv[2]
    profile = PROFILE
    if len(sys.argv) > 3:
        if len(sys.argv) == 4:
            index = "ABCDE".index(sys.argv[3].upper())
            if index < 0:
                print("Invalid profile")
                return
            profile = [0.0] * 5
            profile[index] = 1.0
        else:
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
