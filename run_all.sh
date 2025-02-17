python classification.py --save --augmenter=AugmenterTPT --loss=defaultTPT --n_aug=63 --n_patches=0 --num_workers=12;
python classification.py --save --augmenter=AugmenterTPT --loss=defaultTPT --n_aug=63 --n_patches=0 --augmix --num_workers=12 --severity=2;
python classification.py --save --augmenter=PatchAugmenter --loss=defaultTPT --n_aug=0 --n_patches=64 --num_workers=12;
python classification.py --save --augmenter=PatchAugmenter --loss=defaultTPT --n_aug=4 --n_patches=16 --augmix --num_workers=12 --severity=2;
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss1 --n_aug=4 --n_patches=16 --augmix --num_workers=12 --severity=2;
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss2 --n_aug=4 --n_patches=16 --augmix --num_workers=12 --severity=2;
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss3 --n_aug=4 --n_patches=16 --augmix --num_workers=12 --severity=2;
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss4 --n_aug=4 --n_patches=16 --augmix --num_workers=12 --severity=2;
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss4 --n_aug=2 --n_patches=32 --augmix --num_workers=12 --severity=2;
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss5 --n_aug=4 --n_patches=16 --augmix --num_workers=12 --severity=2;
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss6 --n_aug=2 --n_patches=32 --augmix --num_workers=12 --alpha_exponential_weightening=2 --severity=2;
python classification.py --save --augmenter=AugmenterTPT --loss=crossentropy_hard1 --n_aug=63 --n_patches=0 --num_workers=12;
python classification.py --save --augmenter=AugmenterTPT --loss=crossentropy_hard5 --n_aug=63 --n_patches=0 --num_workers=12;
python classification.py --save --augmenter=AugmenterTPT --loss=crossentropy_soft --n_aug=63 --n_patches=0 --num_workers=12;
python classification.py --save --augmenter=PatchAugmenter --loss=defaultTPT --n_aug=0 --n_patches=64 --num_workers=12 --overlap=0.20;
python classification.py --save --augmenter=PatchAugmenter --loss=defaultTPT --n_aug=0 --n_patches=64 --num_workers=12 --overlap=0.40;
python classification.py --save --augmenter=PatchAugmenter --loss=defaultTPT --n_aug=2 --n_patches=32 --num_workers=12 --overlap=0.20 --severity=2;
python classification.py --save --augmenter=PatchAugmenter --loss=defaultTPT --n_aug=2 --n_patches=32 --num_workers=12 --overlap=0.40 --severity=2;
python classification.py --save --augmenter=PatchAugmenter --loss=crossentropy_hard5 --n_aug=0 --n_patches=64 --num_workers=12 --overlap=0.20

