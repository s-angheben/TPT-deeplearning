# TPT-deeplearning

## Usage
```bash
usage: classification.py [-h] [--imagenet_a_path IMAGENET_A_PATH] [--coop_weight_path COOP_WEIGHT_PATH] [--n_aug N_AUG] [--n_patches N_PATCHES] [--batch_size BATCH_SIZE] [--arch ARCH] [--device DEVICE]
                         [--learning_rate LEARNING_RATE] [--n_ctx N_CTX] [--ctx_init CTX_INIT] [--class_token_position CLASS_TOKEN_POSITION] [--csc] [--run_name RUN_NAME] [--augmenter AUGMENTER] [--loss LOSS]
                         [--augmix] [--no-augmix] [--severity SEVERITY] [--num_workers NUM_WORKERS] [--save] [--no-save] [--reduced_size REDUCED_SIZE] [--dataset_shuffle] [--no-dataset_shuffle] [--save_imgs]
                         [--no-save_imgs]

TPT-deeplearning, coop and TPT-next

options:
  -h, --help            show this help message and exit
  --imagenet_a_path IMAGENET_A_PATH
                        Path to ImageNet-A dataset
  --coop_weight_path COOP_WEIGHT_PATH
                        Path to pre-trained CoOp weights
  --n_aug N_AUG         Number of augmentations
  --n_patches N_PATCHES
                        Number of patches for patch augmenter
  --batch_size BATCH_SIZE
                        Batch size
  --arch ARCH           Model architecture
  --device DEVICE       Device to use, e.g., 'cuda:0' or 'cpu'
  --learning_rate LEARNING_RATE
                        Learning rate
  --n_ctx N_CTX         Number of context tokens
  --ctx_init CTX_INIT   Context token initialization
  --class_token_position CLASS_TOKEN_POSITION
                        Class token position ('end' or 'start')
  --csc                 Enable class-specific context (CSC)
  --run_name RUN_NAME   Custom name for TensorBoard run
  --augmenter AUGMENTER
                        Select the agumenter: AugmenterTPT, PatchAugmenter
  --loss LOSS           Select the loss: defaultTPT, patch_loss1, patch_loss2, patch_loss3, patch_loss4
  --augmix              Enable augmix
  --no-augmix           Disable augmix
  --severity SEVERITY   Augmix severity
  --num_workers NUM_WORKERS
                        number of workers
  --save                Enable save to TensorBoard
  --no-save             Disable save to TensorBoard
  --reduced_size REDUCED_SIZE
                        number of data sample
  --dataset_shuffle     Shuffle the dataset
  --no-dataset_shuffle  Don't shuffle the dataset
  --save_imgs           Enable saving images
  --no-save_imgs        Disable saving images
```

To run tensorboard:
```bash
tensorboard --logdir=runs
```

## RUNS:

- base TPT + coop (augumentation = randomcrop)
```bash
python classification.py --save --augmenter=AugmenterTPT --loss=defaultTPT --n_aug=63 --n_patches=0 --num_workers=4
```

- base TPT + coop (augumentation = randomcrop + augmix)
```bash
python classification.py --save --augmenter=AugmenterTPT --loss=defaultTPT --n_aug=63 --n_patches=0 --augmix --num_workers=4
```

- simple patches: TPT + coop (augumentation = patches)
```bash
python classification.py --save --augmenter=PatchAugmenter --loss=defaultTPT --n_aug=0 --n_patches=64 --num_workers=4
```

- simple patches: TPT + coop (augumentation = patches + augmix)
```bash
python classification.py --save --augmenter=PatchAugmenter --loss=defaultTPT --n_aug=4 --n_patches=16 --augmix --num_workers=4
```

- patches1: TPT + coop (augumentation = patches + augmix)
```bash
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss1 --n_aug=4 --n_patches=16 --augmix --num_workers=4
```

- patches2: TPT + coop (augumentation = patches + augmix)
```bash
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss2 --n_aug=4 --n_patches=16 --augmix --num_workers=4
```

- patches3: TPT + coop (augumentation = patches + augmix)
```bash
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss3 --n_aug=4 --n_patches=16 --augmix --num_workers=4
```

- patches4: TPT + coop (augumentation = patches + augmix)
```bash
python classification.py --save --augmenter=PatchAugmenter --loss=patch_loss4 --n_aug=4 --n_patches=16 --augmix --num_workers=4
```

