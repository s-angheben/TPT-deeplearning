from data.dataloader import ImageNetA, get_dataloader
from data.datautils import AugmenterTPT
from model.custom_clip import get_coop
from utils.utils import set_random_seed

import torch.backends.cudnn as cudnn
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import sys


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[
        : int(batch_entropy.size()[0] * top)
    ]
    return logits[idx]


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(
        dim=-1, keepdim=True
    )  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(
        logits.shape[0]
    )  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def defaultTPT_loss(output, args, selection_p=0.1):
    output = select_confident_samples(output, selection_p)
    return avg_entropy(output)


def reshape_output_patches(output, n_aug):
    return output.view(-1, n_aug + 1, output.shape[-1])


def test_time_tuning(model, inputs, optimizer, scaler, args, tta_step=1):
    for _ in range(tta_step):
        with torch.cuda.amp.autocast():
            output = model(inputs)
            loss = args.loss(output, args)

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

    return


def test_time_adapt_eval(
    dataloader, model, optimizer, optim_state, scaler, writer, device, args
):
    samples = 0.0
    cumulative_accuracy_base = 0.0
    cumulative_accuracy_tpt = 0.0

    model.eval()
    with torch.no_grad():
        model.reset()

    print("Test Time Evaluation")

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (imgs, target) in progress_bar:
        images = torch.cat(imgs, dim=0).to(device)
        # images = torch.cat(imgs[1:], dim=0).to(device)  # don't consider original image
        orig_img = imgs[0].to(device)
        target = target.to(device)

        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(orig_img)
        pred_base_conf, pred_base_class = torch.softmax(output, dim=1).max(1)

        test_time_tuning(model, images, optimizer, scaler, args)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(orig_img)
        pred_tpt_conf, pred_tpt_class = torch.softmax(output, dim=1).max(1)

        cumulative_accuracy_base += pred_base_class.eq(target).sum().item()
        cumulative_accuracy_tpt += pred_tpt_class.eq(target).sum().item()
        samples += 1

        curr_base_acc = (cumulative_accuracy_base / samples) * 100
        curr_TPTcoop_acc = (cumulative_accuracy_tpt / samples) * 100

        writer.add_scalar("confidence/Base", pred_base_conf * 100, i)
        writer.add_scalar("confidence/TPT_coop", pred_tpt_conf * 100, i)
        writer.add_scalar(
            "TPT_coop/improvement",
            pred_tpt_class.eq(target).sum().item()
            - pred_base_class.eq(target).sum().item(),
            i,
        )
        writer.add_scalar("Accuracy/Base", curr_base_acc, i)
        writer.add_scalar("Accuracy/TPT_coop", curr_TPTcoop_acc, i)

        progress_bar.set_postfix(
            {
                "Base Acc": f"{curr_base_acc:.2f}%",
                "TPT Acc": f"{curr_TPTcoop_acc:.2f}%",
            }
        )

    return cumulative_accuracy_tpt / samples * 100


def generate_run_name(args):
    if args.save:
        config_name = f"augmenter={args.augmenter}_loss={args.loss}_naug={args.n_aug}_npatch={args.n_patches}_augmix={args.augmix}_severity={args.severity}"
        return f"{args.run_name}_{config_name}" if args.run_name else f"{config_name}"
    else:
        return "tmp"


def parse_loss(args):
    if args.loss == "defaultTPT":
        args.loss = defaultTPT_loss
    else:
        exit("Loss not valid")


def parse_augmenter(args):
    if args.augmenter == "AugmenterTPT":
        args.augmenter = AugmenterTPT(args.n_aug, args.augmix, args.severity)
    else:
        exit("Augmenter not valid")


def main():
    args = parser.parse_args()
    run_name = generate_run_name(args)

    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    writer.add_text("Config", json.dumps(vars(args), indent=4))
    parse_augmenter(args)
    parse_loss(args)

    set_random_seed(1234)

    classnames = ImageNetA.classnames
    dataset = ImageNetA(args.imagenet_a_path, transform=args.augmenter)
    dataloader = get_dataloader(
        dataset,
        args.batch_size,
        shuffle=False,
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
    parser.add_argument("--n_aug", type=int, default=64, help="Number of augmentations")
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
        "--augmenter", type=str, default="AugmenterTPT", help="Select the agumenter"
    )
    parser.add_argument(
        "--loss", type=str, default="defaultTPT", help="Select the loss"
    )
    parser.add_argument("--augmix", type=bool, default=False, help="Enable augmix")
    parser.add_argument("--severity", type=int, default=1, help="Augmix severity")
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers")
    parser.add_argument(
        "--save", type=bool, default=False, help="enable save to tensorboard"
    )
    parser.add_argument(
        "--reduced_size", type=int, default=None, help="number of data sample"
    )

    main()
