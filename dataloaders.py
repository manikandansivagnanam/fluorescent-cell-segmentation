from pathlib import Path

import albumentations as A
import pandas as pd
from torch.utils.data import DataLoader

from dataset import NeuronSegmentationDataset


def get_transforms(img_size=256):
    train_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
    ])

    val_transforms = A.Compose([
        A.Resize(img_size, img_size),
    ])

    return train_transforms, val_transforms


def get_dataframes(splits_csv="data/folds/splits.csv", fold=0):
    df = pd.read_csv(splits_csv)

    train_df = df[(df["split"] == "trainval") & (df["fold"] != fold)].copy().reset_index(drop=True)
    val_df = df[(df["split"] == "trainval") & (df["fold"] == fold)].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def get_datasets(
    fold=0,
    img_size=256,
    image_dir="data/processed/images",
    mask_dir="data/processed/masks",
):
    train_df, val_df, test_df = get_dataframes(fold=fold)
    train_tfms, val_tfms = get_transforms(img_size=img_size)

    train_dataset = NeuronSegmentationDataset(
        dataframe=train_df,
        image_dir=image_dir,
        mask_dir=mask_dir,
        transforms=train_tfms,
        return_metadata=True,
    )

    val_dataset = NeuronSegmentationDataset(
        dataframe=val_df,
        image_dir=image_dir,
        mask_dir=mask_dir,
        transforms=val_tfms,
        return_metadata=True,
    )

    test_dataset = NeuronSegmentationDataset(
        dataframe=test_df,
        image_dir=image_dir,
        mask_dir=mask_dir,
        transforms=val_tfms,
        return_metadata=True,
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    fold=0,
    img_size=256,
    batch_size=2,
    num_workers=0,
    image_dir="data/processed/images",
    mask_dir="data/processed/masks",
):
    train_dataset, val_dataset, test_dataset = get_datasets(
        fold=fold,
        img_size=img_size,
        image_dir=image_dir,
        mask_dir=mask_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
