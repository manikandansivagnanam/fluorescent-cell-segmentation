from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

SPLITS_PATH = Path("data/folds/splits.csv")
IMAGE_DIR = Path("data/processed/images")
MASK_DIR = Path("data/processed/masks")


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read {path}")
    return img


def overlay(image, mask):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    out = image_rgb.copy()
    out[mask > 0] = [255, 0, 0]
    return cv2.addWeighted(out, 0.35, image_rgb, 0.65, 0)


def show_rows(df, start=0, count=12):
    subset = df.iloc[start:start+count]

    for i, row in subset.iterrows():
        image_id = row["image_id"]
        image = load_gray(IMAGE_DIR / image_id)
        mask = load_gray(MASK_DIR / image_id)

        print("=" * 80)
        print(f"image_id: {image_id}")
        print(f"channel : {row['channel']}")
        print(f"split   : {row['split']}")
        print(f"fold    : {row['fold']}")
        print(f"mask unique: {np.unique(mask)}")
        print(f"foreground pixels: {(mask > 0).sum()}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")

        axes[2].imshow(overlay(image, mask))
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


def main():
    df = pd.read_csv(SPLITS_PATH)

    # Look at a mix of samples, but start with yellow since that raised concern.
    yellow_df = df[df["channel"] == "yellow"].reset_index(drop=True)
    show_rows(yellow_df, start=0, count=12)


if __name__ == "__main__":
    main()
