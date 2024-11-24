from data.dataloader import ImageNetA, get_dataloader

from data.datautils import PatchAugmenter
from model.custom_clip import get_coop
from utils.utils import set_random_seed

import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
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
    #print(avg_logits.shape)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def selective_entropy_loss(outputs, entropy_threshold=0.5):
    # Step 1: Average class distributions across augmentations for each patch
    avg_logits_per_patch = outputs.logsumexp(dim=1) - np.log(outputs.shape[1])  # Shape: [patch, classes]
    
    # Step 2: Calculate entropy for each patch
    avg_logits_per_patch = torch.clamp(avg_logits_per_patch, min=torch.finfo(avg_logits_per_patch.dtype).min)
    entropy_per_patch = -(avg_logits_per_patch * torch.exp(avg_logits_per_patch)).sum(dim=-1)  # Shape: [patch]
    
    # Step 3: Define the loss conditionally based on entropy threshold
    # 3a: Force consistency between patches if entropy across patches is high
    if entropy_per_patch.mean() > entropy_threshold:
        # Minimize variance or KL divergence between patches
        consistency_loss = 0
        for i in range(avg_logits_per_patch.shape[0] - 1):
            for j in range(i + 1, avg_logits_per_patch.shape[0]):
                consistency_loss += torch.nn.functional.kl_div(
                    avg_logits_per_patch[i].log_softmax(dim=0), avg_logits_per_patch[j].softmax(dim=0), reduction="batchmean"
                )
        consistency_loss /= avg_logits_per_patch.shape[0] * (avg_logits_per_patch.shape[0] - 1) / 2
        return consistency_loss
    
    # 3b: Otherwise, target only patches with high entropy
    else:
        high_entropy_patches = entropy_per_patch > entropy_threshold
        selective_entropy_loss = entropy_per_patch[high_entropy_patches].mean()
        return selective_entropy_loss

# 5 4 200
# 3 augmentations 
# 4 patches

def select_most_frequent_class(logits):
    probabilities = logits.softmax(dim=-1)
    predicted_classes = probabilities.argmax(dim=2)
    most_frequent_classes = torch.mode(predicted_classes, dim=1).values
    return most_frequent_classes

def weighted_most_frequent_class(entropy, classes):
    weights = 1 / (entropy + 1e-6)
    weighted_counts = {}
    for i, cls in enumerate(classes):
        if cls.item() not in weighted_counts:
            weighted_counts[cls.item()] = 0
        weighted_counts[cls.item()] += weights[i].item()
    most_frequent_class = max(weighted_counts, key=weighted_counts.get)
    return torch.tensor(most_frequent_class).to(entropy.device)

def weighted_class_distribution(entropy, classes, n_classes):
    weights = 1 / (entropy + 1e-6)
    
    weighted_counts = {}
    for i, cls in enumerate(classes):
        if cls.item() not in weighted_counts:
            weighted_counts[cls.item()] = 0
        weighted_counts[cls.item()] += weights[i].item()
    
    total_weight = sum(weighted_counts.values())
    distribution = torch.zeros(n_classes, device=entropy.device)
    
    for cls, weight in weighted_counts.items():
        distribution[cls] = weight / total_weight

    return distribution

def patch_loss(outputs, n_patches, patch_selection_p=0.9):
    selected_outputs = []

    output_original_image = outputs[0][0]

    patch_entropy = []
    for i in range(0, n_patches * n_patches + 1): 
        selected_output = select_confident_samples(outputs[i], patch_selection_p)
        patch_entropy.append(avg_entropy(selected_output))
        
        selected_outputs.append(selected_output)  # Append each selected output to the list
        '''
        for s in selected_output:
            selected_class = torch.softmax(s, dim=0).max(0).indices.item()
            print(selected_class)
        '''
    
    # 1 Cross entropy loss between the original image and the weighted probability distribution 
    '''
    all_selected_output = torch.stack(selected_outputs, dim=0)
    patch_entropy = torch.stack(patch_entropy, dim=0)
    patch_classes = select_most_frequent_class(all_selected_output) # Calculate the most frequent class for each patch
    # Calculate the weighted class distribution based on occorrencies in order to use it as the target distribution
    target_distribution = weighted_class_distribution(patch_entropy[1:], patch_classes[1:], output_original_image.shape[-1])  
    # Calculate the cross entropy loss between the original image and the target probability distribution
    log_probs = F.log_softmax(output_original_image, dim=-1)
    loss = -(target_distribution * log_probs).sum()
    '''

    # 2 Cross entropy loss between the original image and the most frequent class
    '''
    all_selected_output = torch.stack(selected_outputs, dim=0)
    patch_entropy = torch.stack(patch_entropy, dim=0)
    patch_classes = select_most_frequent_class(all_selected_output) # Calculate the most frequent class for each patch
    # Calculate the most frequent class for the patches weighted by their entropy
    most_frequent_class = weighted_most_frequent_class(patch_entropy[1:], patch_classes[1:])  
    # Calculate the cross entropy loss between the original image and the most frequent class
    loss = F.cross_entropy(output_original_image.unsqueeze(0), most_frequent_class.unsqueeze(0))
    '''
    # 3 Give a weight to each patch based on its entropy
    '''
    patch_entropy = torch.stack(patch_entropy, dim=0)
    epsilon = 1e-6
    weights = 1 / (patch_entropy + epsilon)
    weights /= weights.sum() #normalization
    #print("Weights", weights)
    #print("Entropy patches", patch_entropy)
    #print("Weights", weights)
    weighted_entropy = (patch_entropy * weights).sum()
    #print("Weighted entropy", weighted_entropy)
    
    loss = weighted_entropy
    '''
    # 4 Use the loss of the patch with the lowest entropy
    patch_entropy = torch.stack(patch_entropy, dim=0)
    loss = patch_entropy.min()
    
    return loss


# n_aug = 2, n_patches = 2, 
# torch.Size([13, 200]): [orig_img, patch1, patch1_aug1, patch1_aug2, patch1_aug3, ... , patch4_aug1, patch4_aug2, patch4_aug3]
# return 
# torch.Size([5, 3, 200])
def reshape_output_patches(output, n_aug):
    return output.view(-1, n_aug+1, output.shape[-1])
    # a = tt.view(-1, n_patches+1, 200)
    # b = tt.view(-1, n_aug+1, 200)
    # assert(torch.equal(a, b))

def test_time_tuning(
    model, n_aug, n_patches, inputs, optimizer, scaler, tta_step=1, selection_p=0.2
):
    for _ in range(tta_step):
        with torch.cuda.amp.autocast():
            output = model(inputs)
            output = reshape_output_patches(output, n_aug)
            loss = patch_loss(output, n_patches, selection_p)
            #loss = selective_entropy_loss(output, n_aug)
            # output = select_confident_samples(output, selection_p)
            #loss = avg_entropy(output)
            

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

    return


def compute_statistics(statistics):
    statistics_global = {
        "tpt_improved_samples": 0,
        "tpt_worsened_samples": 0,
        "n_samples": 0,
    }

    for i in range(200):
        if statistics[i]["n_samples"] != 0:
            # global statistics
            statistics_global["tpt_improved_samples"] += statistics[i][
                "tpt_improved_samples"
            ]
            statistics_global["tpt_worsened_samples"] += statistics[i][
                "tpt_worsened_samples"
            ]
            statistics_global["n_samples"] += statistics[i]["n_samples"]

            # class statistics
            statistics[i]["tpt_improved_samples"] /= statistics[i]["n_samples"]
            statistics[i]["tpt_worsened_samples"] /= statistics[i]["n_samples"]
            print(
                f"Class {i}: Improved {statistics[i]['tpt_improved_samples']:.2f}, Worsened {statistics[i]['tpt_worsened_samples']:.2f}, Samples {statistics[i]['n_samples']}"
            )

    print(
        f"Global: Improved {statistics_global['tpt_improved_samples']/statistics_global['n_samples']:.4f}, Worsened {statistics_global['tpt_worsened_samples']/statistics_global['n_samples']:.4f}, Samples {statistics_global['n_samples']}"
    )


def test_time_adapt_eval(
    dataloader, model, n_aug, n_patches, optimizer, optim_state, scaler, writer, device
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

        test_time_tuning(model, n_aug, n_patches, images, optimizer, scaler)

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
    n_aug=15,
    n_patches=4,
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

    augmenter = PatchAugmenter(n_aug=n_aug, n_patches=n_patches)

    dataset = ImageNetA(ImageNetA_path, transform=augmenter)
    dataloader = get_dataloader(
        dataset, batch_size, shuffle=True, reduced_size=None, num_workers=8
    )

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
        dataloader,
        model,
        n_aug,
        n_patches,
        optimizer,
        optim_state,
        scaler,
        writer,
        device,
    )
    print(result)

    writer.close()


if __name__ == "__main__":
    main()
