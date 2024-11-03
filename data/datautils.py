from torchvision.transforms import v2
import torchvision.transforms.functional as F

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_clip_preprocess():
    n_px = 224
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_patches(image, clip_preprocess, n):
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

    return images

class PatchAugmenter(object):
    def __init__(self, n_aug=2, n_patches=16):
        self.preprocess = get_clip_preprocess()
        self.n_aug = n_aug
        self.augmix = v2.AugMix()
        self.crop_patches = get_patches
        self.n_patches = n_patches

    def __call__(self, x):
        img = self.preprocess(x)
        img_orig_aug = [self.augmix(img) for _ in range(self.n_aug)]
        patches = self.crop_patches(x, self.preprocess, self.n_patches)
        patches_augm = [augmented_patch for patch in patches for augmented_patch in [patch] + [self.augmix(patch) for _ in range(self.n_aug)]]
        return [img] + img_orig_aug + patches_augm


# img_orig + patch_orig + patch_augm
# img_orig + [[patch1_orig, patch_1augm, ..], ..., [patchn_orig, patch_naugm, ..]]
# img_orig + [patch1_orig, patch_1augm, ..], ..., [patchn_orig, patch_naugm, ..]


# image pathces 
# # # # # 
# # # # # 
# # # # # 
# # # # #

#prompt
# # # #} n 

## 1)
# similarity between patches
# if high -> get class
# if low  -> select for each patch the class with lower entropy

## 2)
# use external model for augumentation