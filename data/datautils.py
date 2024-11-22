import torch
import torchvision.transforms as transforms
import numpy as np
import math

BICUBIC = transforms.InterpolationMode.BICUBIC


class IdentityTransform(torch.nn.Module):
    def forward(self, x):
        return x


# AugMix Transforms
def get_base_transform():
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
        ]
    )


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_preprocess():
    return transforms.Compose(
        [
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def get_clip_preprocess():
    base_transform = get_base_transform()
    preprocess = get_preprocess()
    return transforms.Compose(base_transform.transforms + preprocess.transforms)


def get_preaugment():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    )


## applies clip_preprocess to the original images and
## random resized and horizontal flip + augmix for the rest of the images
class AugmenterTPT(object):
    def __init__(self, n_aug=2, augmix=False, severity=1):
        self.n_aug = n_aug
        self.base_transform = get_base_transform()
        self.preprocess = get_preprocess()
        self.preaugment = get_preaugment()
        self.augmix = (
            transforms.AugMix(severity=severity) if augmix else IdentityTransform()
        )

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        augmented = [
            self.augmix(self.preprocess(self.preaugment(x))) for _ in range(self.n_aug)
        ]
        return [image] + augmented


#################################################


def crop_patches(image, clip_preprocess, n):
    n = math.floor(math.sqrt(n))
    width, height = image.size

    part_width = width // n
    part_height = height // n
    images = []

    for i in range(n):
        for j in range(n):
            left = j * part_width
            upper = i * part_height
            right = (j + 1) * part_width if j < n - 1 else width
            lower = (i + 1) * part_height if i < n - 1 else height

            cropped_image = image.crop((left, upper, right, lower))
            images.append(clip_preprocess(cropped_image))

    return images


class PatchAugmenter(object):
    def __init__(self, n_aug=2, n_patches=16, severity=1):
        self.n_aug = n_aug
        self.n_patches = n_patches
        self.clip_preprocess = get_clip_preprocess()
        self.augmix = transforms.AugMix(severity=severity)
        self.crop_patches = crop_patches

    def __call__(self, x):
        img = self.clip_preprocess(x)
        img_orig_aug = [self.augmix(img) for _ in range(self.n_aug)]
        patches = self.crop_patches(x, self.clip_preprocess, self.n_patches)
        patches_augm = [
            augmented_patch
            for patch in patches
            for augmented_patch in [patch]
            + [self.augmix(patch) for _ in range(self.n_aug)]
        ]
        return [img] + img_orig_aug + patches_augm
