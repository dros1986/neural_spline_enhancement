import torch
import numpy as np
import random
import sys
sys.path.append("..")
from Dataset5 import Dataset
from NeuralSpline5 import NeuralSpline
import ptcolor


MODEL = "../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_abcde_emb_best.pth"

RAWDIR = "/mnt/data/dataset/fivek/new-users/fivek-test100-raw"
EXPDIR = "/mnt/data/dataset/fivek/new-users/"
LIST = "/mnt/data/dataset/fivek/new-users/list.txt"


def _load_images(nimages, expert, raw_dir, expert_dir, list_file):
    dataset = Dataset(raw_dir, [expert_dir], list_file, ["user-" + expert.upper()], False)
    raw = torch.zeros(nimages, 3, 256, 256)
    target = torch.zeros(nimages, 3, 256, 256)
    n = len(dataset)
    for i, index in enumerate(random.sample(range(n), nimages)):
        r, t, _ = dataset[index]
        raw[i, ...] = r
        target[i, ...] = t
    return raw, target


def vote_profile(expert, vote_file, n):
    e = "fghij".index(expert.lower())
    votes = np.loadtxt(vote_file)
    cumv = np.cumsum(votes[e, :])
    profile = np.zeros(5)
    for _ in range(n):
        c = np.random.randint(cumv[-1])
        k = (cumv > c).nonzero()[0][0]
        profile[k] += 1
    profile /= profile.sum()
    return profile


def fit_profile(model, expert, nimages, list_file, raw_dir, expert_dir,
                iterations=100, batch_size=10, learning_rate=0.1,
                verbose=False):
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    mask = [0.0 if x == expert else 1.0 for x in "abcde"]
    mask = torch.tensor(mask).to(device)
    who = [0.0 if x == expert else 0.25 for x in "abcde"]
    who = torch.tensor(who).to(device)
    who.requires_grad = True
    raw_cpu, target_cpu = _load_images(nimages, expert, raw_dir, expert_dir, list_file)
    if verbose:
        print("Load", raw_cpu.size(0), "images")
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
    # python3 learnprofile.py  STRATEGY TARGET_USER N_IMAGES
    expert = "f"
    nimages = 100
    strategy = "vote"
    if len(sys.argv) > 1:
        strategy = sys.argv[1].lower()  # vote or fit
    if len(sys.argv) > 2:
        expert = sys.argv[2].lower()
    if len(sys.argv) > 3:
        nimages = int(sys.argv[3])

    if strategy == "fit":
        model = NeuralSpline(10, 8, 1)
        model_file = MODEL
        data = torch.load(model_file)
        model.load_state_dict(data["state_dict"])
        model.cuda()
        profile = fit_profile(model, expert, nimages, LIST, RAWDIR, EXPDIR, verbose=True)
    elif strategy == "vote":
        profile = vote_profile(expert, "votes.txt", nimages)
    else:
        print("Strategy must be 'vote' or 'fit'")
        profile = [float("nan")] * 5
    print()
    print(*profile)
    # Map on the bidimensional manifold by using the 5 experts
    experts = [
        [-0.3469, -0.1896],
        [0.1869, -0.3367],
        [-0.1448,  0.1571],
        [0.1845,  0.0535],
        [0.2760,  0.3392]
        ]
    ps = [sum(e[i] * p for e, p in zip(experts, profile)) for i in (0,1)]
    print(*ps)


if __name__ == "__main__":
    main()
