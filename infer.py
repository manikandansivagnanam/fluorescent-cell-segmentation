import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as A

from dataset import NeuronSegmentationDataset
from model import get_model
from metrics import dice_score, iou_score, precision_score, recall_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    output_pred_dir = Path("outputs/predictions")
    output_metric_dir = Path("outputs/metrics")
    output_pred_dir.mkdir(parents=True, exist_ok=True)
    output_metric_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/folds/splits.csv")

    if args.split == "val":
        eval_df = df[(df["split"] == "trainval") & (df["fold"] == args.fold)].copy().reset_index(drop=True)
    elif args.split == "test":
        eval_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    else:
        raise ValueError("split must be 'val' or 'test'")

    transforms = A.Compose([
        A.Resize(args.img_size, args.img_size),
    ])

    dataset = NeuronSegmentationDataset(
        dataframe=eval_df,
        image_dir="data/processed/images",
        mask_dir="data/processed/masks",
        transforms=transforms,
        return_metadata=True,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = get_model(model_name=args.model, encoder_name=args.encoder).to(DEVICE)
    ckpt_path = Path(f"outputs/checkpoints/{args.model}_{args.encoder}_fold{args.fold}.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    save_dir = output_pred_dir / f"{args.model}_{args.encoder}_{args.split}_fold{args.fold}"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_probs = []
    all_masks = []

    with torch.no_grad():
        for images, masks, metadata in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_masks.append(masks.cpu())

            pred = (probs[0, 0].cpu().numpy() > args.threshold).astype(np.uint8) * 255
            image_id = metadata["image_id"][0]
            cv2.imwrite(str(save_dir / image_id), pred)

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    result = pd.DataFrame([{
        "model": args.model,
        "encoder": args.encoder,
        "fold": args.fold,
        "split": args.split,
        "img_size": args.img_size,
        "threshold": args.threshold,
        "dice": dice_score(all_probs, all_masks, threshold=args.threshold),
        "iou": iou_score(all_probs, all_masks, threshold=args.threshold),
        "precision": precision_score(all_probs, all_masks, threshold=args.threshold),
        "recall": recall_score(all_probs, all_masks, threshold=args.threshold),
    }])

    out_csv = output_metric_dir / f"result_{args.model}_{args.encoder}_{args.split}_fold{args.fold}.csv"
    result.to_csv(out_csv, index=False)

    print(result)
    print("Saved predictions to:", save_dir)
    print("Saved metrics to    :", out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="UnetPlusPlus")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
