from data.dataloader import *
from model.custom_clip import get_coop
from data.datautils import *
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy


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


def test_time_tuning(model, inputs, optimizer, tta_step=1, selection_p=0.1):
    for _ in range(tta_step):
        output = model(inputs)
        output = select_confident_samples(output, selection_p)
        loss = avg_entropy(output)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return

def compute_statistics(statistics):
    for i in range(200):
        if statistics[i]["n_samples"] != 0:
            statistics[i]["tpt_improved_samples"] /= statistics[i]["n_samples"]
            statistics[i]["tpt_worsened_samples"] /= statistics[i]["n_samples"]
            print(f"Class {i}: Improved {statistics[i]['tpt_improved_samples']:.2f}, Worsened {statistics[i]['tpt_worsened_samples']:.2f}, Samples {statistics[i]['n_samples']}")
     

def test_time_adapt_eval(dataloader, model, optimizer, optim_state, device):
    samples = 0.0
    cumulative_accuracy_base = 0.0
    cumulative_accuracy_tpt = 0.0
    model.eval()
    with torch.no_grad():
        model.reset()

    print("Test Time Evaluation")

    statistics = [{"tpt_improved_samples": 0, "tpt_worsened_samples": 0, "n_samples": 0} for _ in range(200)]

    progress_bar = tqdm(enumerate(dataloader), total=500)
    # for i, (imgs, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
    for i, (imgs, target) in progress_bar:
        images = torch.cat(imgs, dim=0).to(device)
        # images = torch.cat(imgs[1:], dim=0).to(device)  # don't consider original image
        orig_img = imgs[0].to(device)
        target = target.to(device)

        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        _, predicted_base = model(orig_img).max(1)

        test_time_tuning(model, images, optimizer)

        with torch.no_grad():
            output = model(orig_img)

        _, predicted_tpt = output.max(1)

        cumulative_accuracy_base += predicted_base.eq(target).sum().item()
        cumulative_accuracy_tpt += predicted_tpt.eq(target).sum().item()
        samples += 1

        statistics[target]["tpt_improved_samples"] += 1 if predicted_base.eq(predicted_tpt).sum().item() != 1 and predicted_tpt.eq(target).sum().item() == 1 else 0
        statistics[target]["tpt_worsened_samples"] += 1 if predicted_base.eq(predicted_tpt).sum().item() != 1 and predicted_base.eq(target).sum().item() == 1 else 0
        statistics[target]["n_samples"] += 1

        progress_bar.set_postfix({
            'Base Acc': f"{(cumulative_accuracy_base / samples) * 100:.2f}%",
            'TPT Acc': f"{cumulative_accuracy_tpt / samples * 100:.2f}%"
        })

        if i == 500:
            break

    compute_statistics(statistics)
    return cumulative_accuracy_tpt / samples * 100


def get_optimizer(model, lr, wd, momentum):
    optimizer = torch.optim.SGD(
        [{"params": model.parameters()}], lr=lr, weight_decay=wd, momentum=momentum
    )

    return optimizer


def get_optimizer2(model, lr, wd, momentum):
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters()}], lr=lr, weight_decay=wd
    )

    return optimizer


def main(
    ImageNetA_path="../Datasets/imagenet-a/",
    coop_weight_path="../model.pth.tar-50",
    n_aug=64 - 1,
    batch_size=1,
    arch="RN50",
    device="cuda:0",
    learning_rate=0.03,
    weight_decay=0.0005,
    momentum=0.9,
    n_ctx=4,
    ctx_init="",
    class_token_position="end",
    csc=False,
):
    classnames = ImageNetA.classnames

    augmenter = Augmenter(n_aug=n_aug)
    dataset = ImageNetA(ImageNetA_path, transform=augmenter)
    dataloader = get_dataloader(dataset, batch_size)

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
    # optimizer = torch.optim.AdamW(trainable_param, learning_rate)
    optimizer = get_optimizer(model, learning_rate, weight_decay, momentum)
    optim_state = deepcopy(optimizer.state_dict())

    cudnn.benchmark = True
    model.reset_classnames(classnames, arch)

    result = test_time_adapt_eval(dataloader, model, optimizer, optim_state, device)
    print(result)


if __name__ == "__main__":
    main()
