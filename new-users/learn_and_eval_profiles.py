#!/usr/bin/env python3

import torch
import numpy as np
import sys
sys.path.append("..")
from NeuralSpline5 import NeuralSpline
import evalprofile
import learnprofile
import functools
import random


TARGET_EXP = "user-F"

MODEL = "../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_abcde_emb_best.pth"
RAWDIR = "/mnt/data/dataset/fivek/new-users/fivek-test100-raw"
EXPDIR = "/mnt/data/dataset/fivek/new-users/"
IMAGE_LIST = "/mnt/data/dataset/fivek/new-users/list.txt"
VOTES_FILE = "votes.txt"

REPS = 100
NIMAGES = [1, 3, 10, 30, 90]


def run_simulations(model, expert, n, strategy, f):
    print_ = functools.partial(print, file=f, flush=True)
    print_("  # Samp Profile" + " " * 25 + "DE76    De94   DL")
    print_("-" * 63)
    fmt = "{:3d} {:3d} " + " {:.3f}" * 5 + "  " + " {:.4f}" * 3
    stats = []

    with open(IMAGE_LIST) as f:
        images = [line.strip() for line in f if line.strip()]

    train_list = "train.tmp"
    test_list = "test.tmp"
    for rep in range(1, REPS + 1):
        random.shuffle(images)
        with open(train_list, "w") as f:
            for line in images[:n]:
                print(line, file=f)
        with open(test_list, "w") as f:
            for line in images[n:]:
                print(line, file=f)
        print(expert, n, ": ", rep, "/", REPS, flush=True)
        if strategy == "fit":
            profile = learnprofile.fit_profile(model, expert, n, train_list, RAWDIR, EXPDIR)
        elif strategy == "vote":
            profile = learnprofile.vote_profile(expert, VOTES_FILE, n)
        de76, de94, dl = evalprofile.eval_profile(profile, model, RAWDIR, EXPDIR, test_list, expert)
        stats.append(np.array([*profile, de76, de94, dl]))
        print_(fmt.format(rep, n, *stats[-1]))
    print_("-" * 63)
    stats = np.vstack(stats)
    print_(("AVG" + fmt[5:]).format(n, *stats.mean(0)))
    print_(("MIN" + fmt[5:]).format(n, *stats.min(0)))
    print_(("MAX" + fmt[5:]).format(n, *stats.max(0)))
    print_()


def process_expert(model, expert, strategy, f):
    for n in NIMAGES:
        run_simulations(model, expert, n, strategy, f)


def main():
    model = NeuralSpline(10, 8, 1)
    model_file = MODEL
    data = torch.load(model_file)
    model.load_state_dict(data["state_dict"])
    model.cuda()
    model.eval()

    for exp in "fghij":
        with open("vote-profile-results-{}.txt".format(exp), "wt") as f:
            process_expert(model, exp, "vote", f)
        with open("fit-profile-results-{}.txt".format(exp), "wt") as f:
            process_expert(model, exp, "fit", f)


if __name__ == "__main__":
    main()
