from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

SPLITS_PATH = Path("data/folds/splits.csv")
IMAGE_DIR = Path("data/processed/images")
MASK_DIR = Path("data/processed/masks")


def main():
    df = pd.read_csv(SPLITS_PATH)
    val_df = df[(df["split"] == "trainval") & (df["fold"] == 0)].copy().reset_index(drop=True)

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    for _, row in val_df.iterrows():
        image_id = row["image_id"]

        img = cv2.imread(str(IMAGE_DIR / image_id), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(MASK_DIR / image_id), cv2.IMREAD_GRAYSCALE)

        _, pred = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        pred = (pred > 0).astype(np.uint8).flatten()
        gt = (gt > 0).astype(np.uint8).flatten()

        dice_scores.append(f1_score(gt, pred, zero_division=0))
        iou_scores.append(jaccard_score(gt, pred, zero_division=0))
        precision_scores.append(precision_score(gt, pred, zero_division=0))
        recall_scores.append(recall_score(gt, pred, zero_division=0))

    print("Otsu baseline on validation fold 0")
    print("Dice     :", np.mean(dice_scores))
    print("IoU      :", np.mean(iou_scores))
    print("Precision:", np.mean(precision_scores))
    print("Recall   :", np.mean(recall_scores))


if __name__ == "__main__":
    main()
