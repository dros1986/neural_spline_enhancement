#!/usr/bin/env python3

import numpy as np
import cv2
import sys
import os
import random


EXPERTS = "expA expB expC expD expE".split()
ENCODING = dict((x, n) for (n, x) in enumerate(EXPERTS))


def imread(filename):
    bgr = cv2.imread(filename)
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return lab


def select_closest(images_ref, images_target):
    m = len(images_ref)
    n = len(images_target)
    dists = {}
    ret = []
    for i in range(m):
        for j in range(n):
            dsq = ((images_ref[i] - images_target[j]) ** 2).sum()
            dists[(i, j)] = dsq
    for j in range(n):
        index = min(range(m), key=lambda i: dists[(i, j)])
        ret.append(index)
    return ret


def _main():
    expdir = "/mnt/data/dataset/fivek/siggraph2018/256x256"
    userdir = "/mnt/data/dataset/fivek/new-users"
    listfile = os.path.join(userdir, "list.txt")
    experts = "expA expB expC expD expE".split()
    users = "user-F user-G user-H user-I user-J".split()
    outfile = "votes.txt"
    
    counts = np.zeros([len(users), len(experts)], dtype=np.int32)
    n = 0
    with open(listfile) as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            exp_images = [imread(os.path.join(expdir, e, line_s)) for e in experts]
            user_images = [imread(os.path.join(userdir, u, line_s)) for u in users]

            indices = select_closest(exp_images, user_images)
            for a, b in enumerate(indices):
                counts[a][b] += 1
            n += 1
            if n % 100 == 0:
                print(".", end="", flush=True)
    print()
    np.savetxt(outfile, counts, fmt="%d")
    print(counts)


if __name__ == "__main__":
    _main()
