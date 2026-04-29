from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NeuronSegmentationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        mask_dir: str,
        transforms=None,
        return_metadata: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.df)

    def load_grayscale(self, path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return image

    def normalize_image(self, image: np.ndarray):
        image = image.astype(np.float32)

        p1 = np.percentile(image, 1)
        p99 = np.percentile(image, 99)

        image = np.clip(image, p1, p99)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        return image

    def binarize_mask(self, mask: np.ndarray):
        return (mask > 0).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = row["image_id"]
        image_path = self.image_dir / image_id
        mask_path = self.mask_dir / image_id

        image = self.load_grayscale(image_path)
        mask = self.load_grayscale(mask_path)

        image = self.normalize_image(image)
        mask = self.binarize_mask(mask)

        # Convert grayscale image to 3 channels because pretrained encoders expect RGB-like input.
        image = np.stack([image, image, image], axis=-1)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.return_metadata:
            metadata = {
                "image_id": image_id,
                "channel": row["channel"],
                "split": row["split"],
                "fold": row["fold"],
            }
            return image, mask, metadata

        return image, mask
