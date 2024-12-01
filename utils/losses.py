import torch
import numpy as np
import torch.nn.functional as F


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


def defaultTPT_loss(output, args):
    output = select_confident_samples(output, args.selection_p_all)
    return avg_entropy(output)


def reshape_output_patches(output, args):
    return output.view(-1, args.n_aug + 1, output.shape[-1])


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


def patch_loss1(outputs, args):
    outputs = reshape_output_patches(outputs, args)
    selected_outputs = []

    output_original_image = outputs[0][0]

    patch_entropy = []
    for i in range(0, args.n_patches + 1):
        selected_output = select_confident_samples(outputs[i], args.selection_p_patch)
        patch_entropy.append(avg_entropy(selected_output))

        selected_outputs.append(
            selected_output
        )  # Append each selected output to the list
        """
        for s in selected_output:
            selected_class = torch.softmax(s, dim=0).max(0).indices.item()
            print(selected_class)
        """

    # 1 Cross entropy loss between the original image and the weighted probability distribution

    all_selected_output = torch.stack(selected_outputs, dim=0)
    patch_entropy = torch.stack(patch_entropy, dim=0)
    patch_classes = select_most_frequent_class(
        all_selected_output
    )  # Calculate the most frequent class for each patch
    # Calculate the weighted class distribution based on occorrencies in order to use it as the target distribution
    target_distribution = weighted_class_distribution(
        patch_entropy[1:], patch_classes[1:], output_original_image.shape[-1]
    )
    # Calculate the cross entropy loss between the original image and the target probability distribution
    log_probs = F.log_softmax(output_original_image, dim=-1)
    loss = -(target_distribution * log_probs).sum()

    return loss


def patch_loss2(outputs, args):
    outputs = reshape_output_patches(outputs, args)
    selected_outputs = []

    output_original_image = outputs[0][0]

    patch_entropy = []
    for i in range(0, args.n_patches + 1):
        selected_output = select_confident_samples(outputs[i], args.selection_p_patch)
        patch_entropy.append(avg_entropy(selected_output))

        selected_outputs.append(
            selected_output
        )  # Append each selected output to the list
        """
        for s in selected_output:
            selected_class = torch.softmax(s, dim=0).max(0).indices.item()
            print(selected_class)
        """
    # 2 Cross entropy loss between the original image and the most frequent class

    all_selected_output = torch.stack(selected_outputs, dim=0)
    patch_entropy = torch.stack(patch_entropy, dim=0)
    patch_classes = select_most_frequent_class(
        all_selected_output
    )  # Calculate the most frequent class for each patch
    # Calculate the most frequent class for the patches weighted by their entropy
    most_frequent_class = weighted_most_frequent_class(
        patch_entropy[1:], patch_classes[1:]
    )
    # Calculate the cross entropy loss between the original image and the most frequent class
    loss = F.cross_entropy(
        output_original_image.unsqueeze(0), most_frequent_class.unsqueeze(0)
    )

    return loss


def patch_loss3(outputs, args):
    outputs = reshape_output_patches(outputs, args)
    selected_outputs = []

    output_original_image = outputs[0][0]

    patch_entropy = []
    for i in range(0, args.n_patches + 1):
        selected_output = select_confident_samples(outputs[i], args.selection_p_patch)
        patch_entropy.append(avg_entropy(selected_output))

        selected_outputs.append(
            selected_output
        )  # Append each selected output to the list
        """
        for s in selected_output:
            selected_class = torch.softmax(s, dim=0).max(0).indices.item()
            print(selected_class)
        """
    # 3 Give a weight to each patch based on its entropy
    patch_entropy = torch.stack(patch_entropy, dim=0)
    epsilon = 1e-6
    weights = 1 / (patch_entropy + epsilon)
    weights /= weights.sum()  # normalization
    print("Weights", weights)
    print("Entropy patches", patch_entropy)
    print("Weights", weights)
    weighted_entropy = (patch_entropy * weights).sum()
    print("Weighted entropy", weighted_entropy)

    loss = weighted_entropy

    return loss


def patch_loss4(outputs, args):
    outputs = reshape_output_patches(outputs, args)
    selected_outputs = []

    output_original_image = outputs[0][0]

    patch_entropy = []
    for i in range(0, args.n_patches + 1):
        selected_output = select_confident_samples(outputs[i], args.selection_p_patch)
        patch_entropy.append(avg_entropy(selected_output))

        selected_outputs.append(
            selected_output
        )  # Append each selected output to the list
        """
        for s in selected_output:
            selected_class = torch.softmax(s, dim=0).max(0).indices.item()
            print(selected_class)
        """
    # 4 Use the loss of the patch with the lowest entropy
    patch_entropy = torch.stack(patch_entropy, dim=0)
    loss = patch_entropy.min()

    return loss


"""
compute the mean logit for each patch
compute the entropy for each patch

output prob is the weighted average of the prob of each patch
weighted by the inverse of the corresponding entropy
"""


def patch_loss5(outputs, args):
    epsilon = 1e-6

    output_reshaped = reshape_output_patches(outputs, args)

    mean_output_per_patch = output_reshaped.mean(dim=1)

    mean_logprob_per_patch = mean_output_per_patch.log_softmax(dim=1)
    entropy_per_patch = -(
        mean_logprob_per_patch * torch.exp(mean_logprob_per_patch)
    ).sum(dim=-1)

    weighted_logprob_per_patch = mean_logprob_per_patch * (
        1 / (entropy_per_patch.unsqueeze(dim=1) + epsilon)
    )

    logprob_output = weighted_logprob_per_patch.mean(dim=0).log_softmax(dim=0)
    entropy_loss = -(logprob_output * torch.exp(logprob_output)).sum(dim=0)

    return entropy_loss


"""
compute the mean logit for each patch
compute the entropy for each patch

output prob is the weighted average of the prob of each patch
weighted exponentially by the corresponding entropy
"""


def patch_loss6(outputs, args):
    alpha = args.alpha_exponential_weightening

    output_reshaped = reshape_output_patches(outputs, args)

    mean_output_per_patch = output_reshaped.mean(dim=1)

    mean_logprob_per_patch = mean_output_per_patch.log_softmax(dim=1)
    entropy_per_patch = -(
        mean_logprob_per_patch * torch.exp(mean_logprob_per_patch)
    ).sum(dim=-1)

    exp_weights = torch.exp(-alpha * entropy_per_patch).unsqueeze(dim=1)
    weighted_logprob_per_patch = mean_logprob_per_patch * exp_weights

    logprob_output = weighted_logprob_per_patch.mean(dim=0).log_softmax(dim=0)
    entropy_loss = -(logprob_output * torch.exp(logprob_output)).sum(dim=0)

    return entropy_loss
