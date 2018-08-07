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


def select_closest(images):
    n = len(images)
    dists = {}
    ret = []
    for i in range(n):
        for j in range(i + 1, n):
            dists[(i, j)] = dists[(j, i)] = ((images[i] - images[j]) ** 2).sum()
        dists[(i, i)] = float("inf")
    for i in range(n):
        ret.append(min(range(n), key=lambda j: dists[(i, j)]))
    return ret


def _main():
    basedir = "/mnt/data/dataset/fivek/siggraph2018/256x256"
    listfile = "/mnt/data/dataset/fivek/siggraph2018/train1+2-list.txt"
    experts = "expA expB expC expD expE".split()
    outfile = "votes.txt"
    
    counts = np.zeros([len(experts)] * 2, dtype=np.int32)
    n = 0
    with open(listfile) as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            images = [imread(os.path.join(basedir, e, line_s)) for e in experts]
            indices = select_closest(images)
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
