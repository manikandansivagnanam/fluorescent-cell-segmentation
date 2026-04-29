import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataloaders import get_dataloaders
from model import get_model
from metrics import dice_score, iou_score, precision_score, recall_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, dice_loss_fn, bce_loss_fn):
    model.train()
    running_loss = 0.0

    for images, masks, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = dice_loss_fn(logits, masks) + bce_loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, dice_loss_fn, bce_loss_fn):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_masks = []

    for images, masks, _ in tqdm(loader, desc="Validation", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = dice_loss_fn(logits, masks) + bce_loss_fn(logits, masks)
        probs = torch.sigmoid(logits)

        running_loss += loss.item()
        all_probs.append(probs.cpu())
        all_masks.append(masks.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    return {
        "val_loss": running_loss / len(loader),
        "dice": dice_score(all_probs, all_masks),
        "iou": iou_score(all_probs, all_masks),
        "precision": precision_score(all_probs, all_masks),
        "recall": recall_score(all_probs, all_masks),
    }


def main(args):
    output_ckpt_dir = Path("outputs/checkpoints")
    output_metric_dir = Path("outputs/metrics")
    output_ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_metric_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", DEVICE)
    print("Model:", args.model)
    print("Encoder:", args.encoder)
    print("Fold:", args.fold)

    train_loader, val_loader, _ = get_dataloaders(
        fold=args.fold,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = get_model(model_name=args.model, encoder_name=args.encoder).to(DEVICE)

    dice_loss_fn = smp.losses.DiceLoss(mode="binary")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dice = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, dice_loss_fn, bce_loss_fn)
        val_metrics = validate_one_epoch(model, val_loader, dice_loss_fn, bce_loss_fn)

        row = {
            "epoch": epoch,
            "model": args.model,
            "encoder": args.encoder,
            "fold": args.fold,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(row)

        print(
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"dice={val_metrics['dice']:.4f} | "
            f"iou={val_metrics['iou']:.4f} | "
            f"precision={val_metrics['precision']:.4f} | "
            f"recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print(f"New best Dice: {best_dice:.4f} at epoch {best_epoch}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement} epoch(s). "
                f"Early stopping patience: {args.patience}"
            )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    ckpt_path = output_ckpt_dir / f"{args.model}_{args.encoder}_fold{args.fold}.pth"
    torch.save(best_state, ckpt_path)

    history_df = pd.DataFrame(history)
    history_path = output_metric_dir / f"history_{args.model}_{args.encoder}_fold{args.fold}.csv"
    history_df.to_csv(history_path, index=False)

    print("\nTraining complete.")
    print("Best validation Dice:", round(best_dice, 4))
    print("Best epoch          :", best_epoch)
    print("Saved checkpoint:", ckpt_path)
    print("Saved history   :", history_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="UnetPlusPlus")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    main(args)
