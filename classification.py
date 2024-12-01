from data.dataloader import ImageNetA, get_dataloader
from data.datautils import AugmenterTPT, PatchAugmenter
from model.custom_clip import get_coop
from utils.utils import set_random_seed, MetricsTracker
from utils.losses import (
    defaultTPT_loss,
    patch_loss1,
    patch_loss2,
    patch_loss3,
    patch_loss4,
    patch_loss5,
)

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import sys


def test_time_tuning(model, inputs, optimizer, scaler, args, tta_step=1):
    loss_value = 0.0
    for _ in range(tta_step):
        with torch.cuda.amp.autocast():
            output = model(inputs)
            loss = args.loss(output, args)
            loss_value = loss.item()

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

    return loss_value


def test_time_adapt_eval(
    dataloader, model, optimizer, optim_state, scaler, writer, device, args
):
    metrics = MetricsTracker(args)

    model.eval()
    with torch.no_grad():
        model.reset()

    print("Test Time Evaluation")

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (imgs, target) in progress_bar:
        view_img = imgs[0]
        images = torch.cat(imgs[1:], dim=0).to(device)  # don't consider view image
        orig_img = imgs[1].to(device)
        target = target.to(device)

        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output_base = model(orig_img)

        loss_value = test_time_tuning(model, images, optimizer, scaler, args)
        print(loss_value)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output_tpt = model(orig_img)

        metrics.update(
            i, view_img, output_base, output_tpt, target, loss_value, writer, args
        )

        progress_bar.set_postfix(
            {
                "Base Acc": f"{metrics.get_accuracy_base():.2f}%",
                "TPT Acc": f"{metrics.get_accuracy_tpt():.2f}%",
            }
        )

    metrics.write_info(writer, args)

    return metrics.get_accuracy_tpt()


def generate_run_name(args):
    if args.save:
        config_name = f"size={args.reduced_size if args.reduced_size else 'Full'}_augmenter={args.augmenter}_loss={args.loss}_naug={args.n_aug}_npatch={args.n_patches}_augmix={args.augmix}_severity={args.severity}_lr={args.learning_rate}_spall={args.selection_p_all}_sppat={args.selection_p_patch}"
        return f"{args.run_name}_{config_name}" if args.run_name else f"{config_name}"
    else:
        return "tmp"


### COMPATIBILITY (augmenter - loss)
## AugmenterTPT - defaultTPT, patch_loss1, patch_loss2, patch_loss3, patch_loss4


def parse_loss(args):
    if args.loss == "defaultTPT":
        args.loss = defaultTPT_loss
    elif args.loss == "patch_loss1":
        args.loss = patch_loss1
    elif args.loss == "patch_loss2":
        args.loss = patch_loss2
    elif args.loss == "patch_loss3":
        args.loss = patch_loss3
    elif args.loss == "patch_loss4":
        args.loss = patch_loss4
    elif args.loss == "patch_loss5":
        args.loss = patch_loss5
    else:
        exit("Loss not valid")


def parse_augmenter(args):
    if args.augmenter == "AugmenterTPT":
        args.augmenter = AugmenterTPT(args.n_aug, args.augmix, args.severity)
    elif args.augmenter == "PatchAugmenter":
        args.augmenter = PatchAugmenter(
            args.n_aug, args.n_patches, args.augmix, args.severity
        )
    else:
        exit("Augmenter not valid")


def main():
    args = parser.parse_args()
    run_name = generate_run_name(args)

    print("Config:", json.dumps(vars(args), indent=4))

    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    writer.add_text("Config", json.dumps(vars(args), indent=4))
    parse_augmenter(args)
    parse_loss(args)

    set_random_seed(1234)

    classnames = ImageNetA.classnames
    dataset = ImageNetA(args.imagenet_a_path, transform=args.augmenter)
    args.nclasses = len(classnames)
    args.classnames = classnames
    dataloader = get_dataloader(
        dataset,
        args.batch_size,
        shuffle=args.dataset_shuffle,
        reduced_size=args.reduced_size,
        num_workers=args.num_workers,
    )
    model = get_coop(args.arch, classnames, args.device, args.n_ctx, args.ctx_init)

    print("Use pre-trained soft prompt (CoOp) as initialization")
    pretrained_ctx = torch.load(args.coop_weight_path)["state_dict"]["ctx"]

    with torch.no_grad():
        model.prompt_learner.ctx.copy_(pretrained_ctx)
        model.prompt_learner.ctx_init_state = pretrained_ctx

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    model = model.to(args.device)

    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.learning_rate)
    optim_state = deepcopy(optimizer.state_dict())
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    cudnn.benchmark = True
    model.reset_classnames(classnames, args.arch)

    result = test_time_adapt_eval(
        dataloader, model, optimizer, optim_state, scaler, writer, args.device, args
    )

    print(result)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPT-deeplearning, coop and TPT-next")
    parser.add_argument(
        "--imagenet_a_path",
        type=str,
        default="../Datasets/imagenet-a/",
        help="Path to ImageNet-A dataset",
    )
    parser.add_argument(
        "--coop_weight_path",
        type=str,
        default="../model.pth.tar-50",
        help="Path to pre-trained CoOp weights",
    )
    parser.add_argument("--n_aug", type=int, default=63, help="Number of augmentations")
    parser.add_argument(
        "--n_patches",
        type=int,
        default=0,
        help="Number of patches for patch augmenter",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--arch", type=str, default="RN50", help="Model architecture")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use, e.g., 'cuda:0' or 'cpu'",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-3, help="Learning rate"
    )
    parser.add_argument("--n_ctx", type=int, default=4, help="Number of context tokens")
    parser.add_argument(
        "--ctx_init", type=str, default="", help="Context token initialization"
    )
    parser.add_argument(
        "--class_token_position",
        type=str,
        default="end",
        help="Class token position ('end' or 'start')",
    )
    parser.add_argument(
        "--csc", action="store_true", help="Enable class-specific context (CSC)"
    )
    parser.add_argument(
        "--run_name", type=str, default="", help="Custom name for TensorBoard run"
    )
    parser.add_argument(
        "--augmenter",
        type=str,
        default="AugmenterTPT",
        help="Select the agumenter: AugmenterTPT, PatchAugmenter",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="defaultTPT",
        help="Select the loss: defaultTPT, patch_loss1, patch_loss2, patch_loss3, patch_loss4, patch_loss5",
    )
    parser.add_argument("--augmix", action="store_true", help="Enable augmix")
    parser.add_argument(
        "--no-augmix", action="store_false", dest="augmix", help="Disable augmix"
    )
    parser.add_argument("--severity", type=int, default=1, help="Augmix severity")
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers")
    parser.add_argument(
        "--save", action="store_true", help="Enable save to TensorBoard"
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save",
        help="Disable save to TensorBoard",
    )
    parser.add_argument(
        "--reduced_size", type=int, default=None, help="number of data sample"
    )
    parser.add_argument(
        "--dataset_shuffle", action="store_true", help="Shuffle the dataset"
    )
    parser.add_argument(
        "--no-dataset_shuffle",
        action="store_false",
        dest="dataset_shuffle",
        help="Don't shuffle the dataset",
    )
    parser.add_argument("--save_imgs", action="store_true", help="Enable saving images")
    parser.add_argument(
        "--no-save_imgs",
        action="store_false",
        dest="save_imgs",
        help="Disable saving images",
    )
    parser.add_argument(
        "--selection_p_all", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "--selection_p_patch", type=float, default=0.9, help="Learning rate"
    )
    main()
