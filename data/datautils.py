from torchvision.transforms import v2

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

class Augmenter(object):
    def __init__(self, n_aug=2):
        self.preprocess = get_clip_preprocess()
        self.n_aug = n_aug
        self.augmix = v2.AugMix()

    def __call__(self, x):
        img = self.preprocess(x)
        aug_imgs = [self.augmix(img) for _ in range(self.n_aug)]
        return [img] + aug_imgs


