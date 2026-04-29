from pathlib import Path

import pandas as pd
import torch

from dataset import NeuronSegmentationDataset

SPLITS_PATH = Path("data/folds/splits.csv")
IMAGE_DIR = "data/processed/images"
MASK_DIR = "data/processed/masks"


def main():
    df = pd.read_csv(SPLITS_PATH)

    train_df = df[(df["split"] == "trainval") & (df["fold"] != 0)].copy()
    val_df = df[(df["split"] == "trainval") & (df["fold"] == 0)].copy()
    test_df = df[df["split"] == "test"].copy()

    print("Train samples:", len(train_df))
    print("Val samples  :", len(val_df))
    print("Test samples :", len(test_df))

    dataset = NeuronSegmentationDataset(
        dataframe=train_df,
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transforms=None,
        return_metadata=True,
    )

    image, mask, metadata = dataset[0]

    print("\nSample loaded successfully")
    print("Image tensor shape:", image.shape)
    print("Mask tensor shape :", mask.shape)
    print("Image dtype       :", image.dtype)
    print("Mask dtype        :", mask.dtype)
    print("Image min/max     :", float(image.min()), float(image.max()))
    print("Mask unique       :", torch.unique(mask))
    print("Metadata          :", metadata)


if __name__ == "__main__":
    main()
