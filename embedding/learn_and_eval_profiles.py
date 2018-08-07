#!/usr/bin/env python3

import torch
import numpy as np
import sys
sys.path.append("..")
from NeuralSpline5 import NeuralSpline
import evalprofile
import learnprofile
import functools


MODEL_BASE = "../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_{}_emb_best.pth"

RAWDIR = "/mnt/data/dataset/fivek/siggraph2018/256x256/raw"
EXPDIR = "/mnt/data/dataset/fivek/siggraph2018/256x256"
TRAIN_LIST = "/mnt/data/dataset/fivek/siggraph2018/train1+2-list.txt"
TEST_LIST = "/mnt/data/dataset/fivek/siggraph2018/train1+2-list.txt"
VOTES_FILE = "votes.txt"

REPS = 10
NIMAGES = [1, 3, 10, 30, 100, 300, 1000]


def run_simulations(model, expert, n, strategy, f):
    print_ = functools.partial(print, file=f, flush=True)
    print_("  # Samp Profile" + " " * 25 + "DE76    De94   DL")
    print_("-" * 63)
    fmt = "{:3d} {:3d} " + " {:.3f}" * 5 + "  " + " {:.4f}" * 3
    stats = []

    for rep in range(1, REPS + 1):
        print(expert, n, ": ", rep, "/", REPS, flush=True)
        if strategy == "fit":
            profile = learnprofile.fit_profile(model, expert, n, TRAIN_LIST, RAWDIR, EXPDIR)
        elif strategy == "vote":
            profile = learnprofile.vote_profile(expert, VOTES_FILE, n)
        de76, de94, dl = evalprofile.eval_profile(profile, model, RAWDIR, EXPDIR, TEST_LIST, expert)
        stats.append(np.array([*profile, de76, de94, dl]))
        print_(fmt.format(rep, n, *stats[-1]))
    print_("-" * 63)
    stats = np.vstack(stats)
    print_(("AVG" + fmt[5:]).format(n, *stats.mean(0)))
    print_(("MIN" + fmt[5:]).format(n, *stats.min(0)))
    print_(("MAX" + fmt[5:]).format(n, *stats.max(0)))
    print_()


def process_expert(expert, strategy, f):
    model = NeuralSpline(10, 8, 1)
    others = "abcde".replace(expert, "")
    model_file = MODEL_BASE.format(others)
    data = torch.load(model_file)
    model.load_state_dict(data["state_dict"])
    model.cuda()
    model.eval()
    for n in NIMAGES:
        run_simulations(model, expert, n, strategy, f)


def main():
    for exp in "abcde":
        with open("vote-profile-results-{}.txt".format(exp), "wt") as f:
            process_expert(exp, "vote", f)
        with open("fit-profile-results-{}.txt".format(exp), "wt") as f:
            process_expert(exp, "fit", f)


if __name__ == "__main__":
    main()
