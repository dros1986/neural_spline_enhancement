import torch
import numpy as np
import matplotlib.pyplot as plt


modelfile = "../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_abcde_emb_best.pth"


def main():
    data = torch.load(modelfile, map_location="cpu")
    mat = data["state_dict"]["wl.weight"].numpy()

    avg = mat.mean(axis=1, keepdims=True)
    matC = mat - avg
    cov = np.cov(matC)
    evals, evecs = np.linalg.eig(cov)
    
    plt.scatter(mat[0, :], mat[1, :])
    for i in range(5):
        plt.annotate("ABCDE"[i], (mat[0, i], mat[1, i]))
    plt.axis()
    
    print("mean:", *avg)
    print("first direction:", *evecs[:, 0])
    print("first variance:", evals[0])
    print("second direction:", *evecs[:, 1])
    print("second variance:", evals[1])
    for i in (0, 1):
        plt.plot([0, evals[i] * evecs[i, 0]], [0, evals[i] * evecs[i, 1]], "br"[i] + "-")
    plt.show()
    

if __name__ == "__main__":
    main()


