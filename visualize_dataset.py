from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

META_PATH = Path("data/processed/metadata.csv")
IMAGE_DIR = Path("data/processed/images")
MASK_DIR = Path("data/processed/masks")


def load_grayscale(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def mask_to_binary(mask: np.ndarray):
    return (mask > 0).astype(np.uint8)


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    overlay = image_rgb.copy()

    binary_mask = mask_to_binary(mask)
    overlay[binary_mask == 1] = [255, 0, 0]

    blended = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)
    return blended


def show_samples(df: pd.DataFrame, n: int = 6, seed: int = 42):
    random.seed(seed)
    sample_df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    for i, row in sample_df.iterrows():
        image_id = row["image_id"]
        channel = row["channel"]
        split = row["split"]

        image_path = IMAGE_DIR / image_id
        mask_path = MASK_DIR / image_id

        image = load_grayscale(image_path)
        mask = load_grayscale(mask_path)
        overlay = overlay_mask_on_image(image, mask)

        print("=" * 80)
        print(f"Sample {i + 1}")
        print(f"image_id      : {image_id}")
        print(f"channel       : {channel}")
        print(f"split         : {split}")
        print(f"image shape   : {image.shape}")
        print(f"mask shape    : {mask.shape}")
        print(f"image dtype   : {image.dtype}")
        print(f"mask dtype    : {mask.dtype}")
        print(f"mask min/max  : {mask.min()} / {mask.max()}")
        print(f"mask unique   : {np.unique(mask)[:20]}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


def main():
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {META_PATH}")

    df = pd.read_csv(META_PATH)

    print("Total samples:", len(df))
    print("\nCounts by channel:")
    print(df["channel"].value_counts())

    print("\nCounts by split:")
    print(df["split"].value_counts())

    print("\nShowing random samples...")
    show_samples(df, n=6, seed=42)


if __name__ == "__main__":
    main()
