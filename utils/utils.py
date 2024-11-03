import random
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def show(imgs, label=None):
    if not isinstance(imgs, list):
        imgs = [imgs]

    num_imgs = len(imgs)
    num_cols = min(4, num_imgs)
    num_rows = math.ceil(num_imgs / num_cols)

    fig, axs = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(3 * num_cols, 3 * num_rows)
    )

    if label is not None:
        fig.suptitle(f"Label: {label}", fontsize=16)

    if num_imgs > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i].imshow(np.asarray(img))
        axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    for j in range(i + 1, num_rows * num_cols):
        axs[j].axis("off")


def show_patches(imgs, n_aug, n_patches, label=None):
    if imgs[0].size() == torch.Size([1, 3, 224, 224]):
        imgs = [img.squeeze(0) for img in imgs] # remove batch info   

    num_cols = n_aug + 1
    num_rows = n_patches*2 + 1

    fig, axs = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(3 * num_cols, 3 * num_rows)
    )

    if label is not None:
        fig.suptitle(f"Label: {label}", fontsize=16)

    #orig_img = imgs[0]
    #axs[0][0].imshow(np.asarray(F.to_pil_image(orig_img.detach())))

    #imgs = imgs[1:]
    for i in range(num_rows):
        for j in range(num_cols):
            img = imgs[i*num_cols + j]
            img = F.to_pil_image(img)
            axs[i][j].imshow(np.asarray(img))
            axs[i][j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
