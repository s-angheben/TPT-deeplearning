import torch
import torchvision.transforms as transforms
import data.augmix_ops_TPT as augmentations
import numpy as np

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC


# AugMix Transforms
def get_preaugment():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    )


def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class Augmenter(object):
    def __init__(self, n_aug=2, augmix=False, severity=1):
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
            ]
        )
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.n_views = n_aug
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [
            augmix(x, self.preprocess, self.aug_list, self.severity)
            for _ in range(self.n_views)
        ]
        return [image] + views
