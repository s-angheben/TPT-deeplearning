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

def crop_patches(image, preprocess, n=16):
    H, W = image.size
    n = int(n ** 0.5) # number of patches in each dimension
    patch_size = H // n
    patches = []
    
    for i in range(n):
        for j in range(n):
            top = i * patch_size
            left = j * patch_size
            patch = F.crop(image, top, left, patch_size, patch_size)
            patches.append(preprocess(patch))
    
    assert len(patches) == n * n 
    return patches

class PatchAugmenter(object):
    def __init__(self, n_aug=2, n_patches=16):
        self.preprocess = get_clip_preprocess()
        self.n_aug = n_aug
        self.augmix = v2.AugMix()
        self.crop_patches = crop_patches
        self.n_patches = n_patches

    def __call__(self, x):
        img = self.preprocess(x)
        patches = self.crop_patches(x, self.preprocess, self.n_patches)
        patches_augm = [[patch]+[self.augmix(patch) for _ in range(self.n_aug)] for patch in patches]
        print()
        return [img] + [patches_augm]


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