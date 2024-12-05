import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import math

BICUBIC = transforms.InterpolationMode.BICUBIC


class IdentityTransform(torch.nn.Module):
    def forward(self, x):
        return x


class NormalizeImgTransform(torch.nn.Module):
    def forward(self, x):
        return x.float() / 255.0


class DeNormalizeImgTransform(torch.nn.Module):
    def forward(self, x):
        return (x.clamp(0, 1) * 255.0).to(torch.uint8)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


# AugMix Transforms
def get_base_transform():
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.PILToTensor(),
        ]
    )


def get_normalizer():
    return transforms.Compose(
        [
            NormalizeImgTransform(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def get_unnormalizer():
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    return transforms.Compose(
        [
            transforms.Normalize(mean=(-mean / std), std=(1.0 / std)),
            DeNormalizeImgTransform(),
        ]
    )


def get_clip_preprocess():
    base_transform = get_base_transform()
    normalizer = get_normalizer()
    return transforms.Compose(base_transform.transforms + normalizer.transforms)


def get_preaugment():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
        ]
    )


## applies clip_preprocess to the original images and
## random resized and horizontal flip + augmix for the rest of the images
class AugmenterTPT(object):
    def __init__(self, n_aug=2, augmix=False, severity=1):
        self.n_aug = n_aug
        self.clip_preprocess = get_clip_preprocess()
        self.normalize = get_normalizer()
        self.preaugment = get_preaugment()
        self.augmix = (
            transforms.AugMix(severity=severity) if augmix else IdentityTransform()
        )

    def __call__(self, x):
        image = self.clip_preprocess(x)
        augmented = [
            self.normalize(self.augmix(self.preaugment(x))) for _ in range(self.n_aug)
        ]
        return [F.to_tensor(x)] + [image] + augmented


#################################################


def crop_patches(image, n, base_transform):
    if n == 0:
        return []

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
            images.append(base_transform(cropped_image))

    return images


def crop_patches_with_overlap(image, n, base_transform, overlap=0.0):
    if n == 0:
        return []

    n = math.floor(math.sqrt(n))
    width, height = image.size
    part_width = width // n
    part_height = height // n

    # Calculate the overlap in pixels
    overlap_width = int(part_width * overlap)
    overlap_height = int(part_height * overlap)

    images = []

    for i in range(n):
        for j in range(n):
            left = max(j * part_width - overlap_width, 0)
            upper = max(i * part_height - overlap_height, 0)
            right = min((j + 1) * part_width + overlap_width, width)
            lower = min((i + 1) * part_height + overlap_height, height)

            cropped_image = image.crop((left, upper, right, lower))
            images.append(base_transform(cropped_image))

    return images


## n_patches is the total number of patches
class PatchAugmenter(object):
    def __init__(self, n_aug=2, n_patches=16, overlap=0.0, augmix=False, severity=1):
        self.n_aug = n_aug
        self.n_patches = n_patches
        self.overlap = overlap
        self.clip_preprocess = get_clip_preprocess()
        self.base_transform = get_base_transform()
        self.normalize = get_normalizer()
        self.augmix = (
            transforms.AugMix(severity=severity) if augmix else IdentityTransform()
        )
        self.crop_patches = crop_patches_with_overlap

    def __call__(self, x):
        img = self.clip_preprocess(x)
        img_orig_aug = [
            self.normalize(self.augmix(self.base_transform(x)))
            for _ in range(self.n_aug)
        ]

        patches = self.crop_patches(
            x, self.n_patches, self.base_transform, overlap=self.overlap
        )
        patches_augm = [
            augmented_patch
            for patch in patches
            for augmented_patch in [self.normalize(patch)]
            + [self.normalize(self.augmix(patch)) for _ in range(self.n_aug)]
        ]
        return [F.to_tensor(x)] + [img] + img_orig_aug + patches_augm
