from data.dataloader import ImageNetA, get_dataloader

# from data.datautils import Augmenter
from data.datautils_TPT import Augmenter
from model.custom_clip import get_coop
from utils.utils import set_random_seed

import torch.backends.cudnn as cudnn
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter


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


def test_time_tuning(model, inputs, optimizer, scaler, tta_step=1, selection_p=0.1):
    for _ in range(tta_step):
        with torch.cuda.amp.autocast():
            output = model(inputs)
            output = select_confident_samples(output, selection_p)
            loss = avg_entropy(output)

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

    return


def compute_statistics(statistics):
    statistics_global = {"tpt_improved_samples": 0, "tpt_worsened_samples": 0, "n_samples": 0}

    for i in range(200):
        if statistics[i]["n_samples"] != 0:
            #global statistics
            statistics_global["tpt_improved_samples"] += statistics[i]["tpt_improved_samples"]
            statistics_global["tpt_worsened_samples"] += statistics[i]["tpt_worsened_samples"]
            statistics_global["n_samples"] += statistics[i]["n_samples"]
            
            #class statistics
            statistics[i]["tpt_improved_samples"] /= statistics[i]["n_samples"]
            statistics[i]["tpt_worsened_samples"] /= statistics[i]["n_samples"]
            print(
                f"Class {i}: Improved {statistics[i]['tpt_improved_samples']:.2f}, Worsened {statistics[i]['tpt_worsened_samples']:.2f}, Samples {statistics[i]['n_samples']}"
            )
    
    print(f"Global: Improved {statistics_global['tpt_improved_samples']/statistics_global['n_samples']:.4f}, Worsened {statistics_global['tpt_worsened_samples']/statistics_global['n_samples']:.4f}, Samples {statistics_global['n_samples']}")

def test_time_adapt_eval(
    dataloader, model, optimizer, optim_state, scaler, writer, device
):
    samples = 0.0
    cumulative_accuracy_base = 0.0
    cumulative_accuracy_tpt = 0.0

    model.eval()
    with torch.no_grad():
        model.reset()

    print("Test Time Evaluation")

    statistics = [
        {"tpt_improved_samples": 0, "tpt_worsened_samples": 0, "n_samples": 0}
        for _ in range(200)
    ]

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

        test_time_tuning(model, images, optimizer, scaler)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(orig_img)
        pred_tpt_conf, pred_tpt_class = torch.softmax(output, dim=1).max(1)

        cumulative_accuracy_base += pred_base_class.eq(target).sum().item()
        cumulative_accuracy_tpt += pred_tpt_class.eq(target).sum().item()
        samples += 1

        statistics[target]["tpt_improved_samples"] += (
            1
            if pred_base_class.eq(pred_tpt_class).sum().item() != 1
            and pred_tpt_class.eq(target).sum().item() == 1
            else 0
        )
        statistics[target]["tpt_worsened_samples"] += (
            1
            if pred_base_class.eq(pred_tpt_class).sum().item() != 1
            and pred_base_class.eq(target).sum().item() == 1
            else 0
        )
        statistics[target]["n_samples"] += 1

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

    compute_statistics(statistics)
    return cumulative_accuracy_tpt / samples * 100


def get_optimizer(model, lr, wd, momentum):
    optimizer = torch.optim.SGD(
        [{"params": model.parameters()}], lr=lr, weight_decay=wd, momentum=momentum
    )

    return optimizer


def main(
    ImageNetA_path="../Datasets/imagenet-a/",
    coop_weight_path="../model.pth.tar-50",
    n_aug=64 - 1,
    batch_size=1,
    arch="RN50",
    device="cuda:0",
    # device="cpu",
    learning_rate=5e-3,
    weight_decay=0.0005,
    momentum=0.9,
    n_ctx=4,
    ctx_init="",
    class_token_position="end",
    csc=False,
):
    set_random_seed(1234)
    writer = SummaryWriter("runs/tpt_coop")

    classnames = ImageNetA.classnames

    augmenter = Augmenter(n_aug=n_aug)
    dataset = ImageNetA(ImageNetA_path, transform=augmenter)
    dataloader = get_dataloader(dataset, batch_size, shuffle=True, reduced_size=None, num_workers=8)

    model = get_coop(arch, classnames, device, n_ctx, ctx_init)
    print("Use pre-trained soft prompt (CoOp) as initialization")
    pretrained_ctx = torch.load(coop_weight_path)["state_dict"]["ctx"]
    with torch.no_grad():
        model.prompt_learner.ctx.copy_(pretrained_ctx)
        model.prompt_learner.ctx_init_state = pretrained_ctx

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    model = model.to(device)

    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, learning_rate)
    optim_state = deepcopy(optimizer.state_dict())
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    cudnn.benchmark = True
    model.reset_classnames(classnames, arch)

    result = test_time_adapt_eval(
        dataloader, model, optimizer, optim_state, scaler, writer, device
    )
    print(result)

    writer.close()


if __name__ == "__main__":
    main()
