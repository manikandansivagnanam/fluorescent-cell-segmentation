import matplotlib.pyplot as plt
import torch

from dataloaders import get_dataloaders


def show_batch(images, masks, metadata, max_items=2):
    batch_size = min(images.size(0), max_items)

    for i in range(batch_size):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        mask = masks[i, 0].cpu().numpy()

        image_gray = image[:, :, 0]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(image_gray, cmap="gray")
        axes[0].set_title(f"Image\n{metadata['image_id'][i]}")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


def main():
    train_loader, val_loader, test_loader = get_dataloaders(
        fold=0,
        img_size=256,
        batch_size=2,
        num_workers=0,
    )

    print("Train batches:", len(train_loader))
    print("Val batches  :", len(val_loader))
    print("Test batches :", len(test_loader))

    batch = next(iter(train_loader))
    images, masks, metadata = batch

    print("\nOne batch loaded successfully")
    print("Image batch shape:", images.shape)
    print("Mask batch shape :", masks.shape)
    print("Image dtype      :", images.dtype)
    print("Mask dtype       :", masks.dtype)
    print("Image min/max    :", float(images.min()), float(images.max()))
    print("Mask unique      :", torch.unique(masks))

    print("\nMetadata keys:", metadata.keys())
    print("First image ids:", metadata["image_id"][:2])

    show_batch(images, masks, metadata, max_items=2)


if __name__ == "__main__":
    main()
