import torch


def dice_score(pred_probs, target, threshold=0.5, eps=1e-7):
    pred = (pred_probs > threshold).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(pred_probs, target, threshold=0.5, eps=1e-7):
    pred = (pred_probs > threshold).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def precision_score(pred_probs, target, threshold=0.5, eps=1e-7):
    pred = (pred_probs > threshold).float()
    target = (target > 0.5).float()

    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))

    precision = (tp + eps) / (tp + fp + eps)
    return precision.mean().item()


def recall_score(pred_probs, target, threshold=0.5, eps=1e-7):
    pred = (pred_probs > threshold).float()
    target = (target > 0.5).float()

    tp = (pred * target).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))

    recall = (tp + eps) / (tp + fn + eps)
    return recall.mean().item()
