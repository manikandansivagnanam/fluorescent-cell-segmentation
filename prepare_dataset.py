import shutil
import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
EXTRACT_DIR = RAW_DIR / "extracted"
PROCESSED_DIR = Path("data/processed")

IMAGE_OUT = PROCESSED_DIR / "images"
MASK_OUT = PROCESSED_DIR / "masks"
META_OUT = PROCESSED_DIR / "metadata.csv"

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_all_zips():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(RAW_DIR.glob("*.zip"))

    if not zip_files:
        print("No zip files found in data/raw")
        return []

    print("Zip files found:")
    for z in zip_files:
        print(f" - {z.name}")

    extracted_roots = []

    for z in zip_files:
        channel_name = z.stem.lower()
        out_dir = EXTRACT_DIR / channel_name
        clean_dir(out_dir)

        print(f"Extracting {z.name} -> {out_dir}")
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(out_dir)

        extracted_roots.append(out_dir)

    return extracted_roots


def resolve_channel_root(extracted_channel_dir: Path) -> Path:
    subdirs = [p for p in extracted_channel_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return extracted_channel_dir


def list_image_files(folder: Path):
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
    )


def get_split_dirs(channel_root: Path, split_name: str):
    split_dir = channel_root / split_name
    image_dir = split_dir / "images"
    mask_dir = split_dir / "ground_truths" / "masks"
    return split_dir, image_dir, mask_dir


def collect_pairs_for_channel(channel_root: Path, channel_name: str):
    rows = []

    for split_name in ["trainval", "test"]:
        split_dir, image_dir, mask_dir = get_split_dirs(channel_root, split_name)

        if not split_dir.exists():
            print(f"[WARN] Missing split folder: {split_dir}")
            continue

        if not image_dir.exists():
            print(f"[WARN] Missing image folder: {image_dir}")
            continue

        if not mask_dir.exists():
            print(f"[WARN] Missing mask folder: {mask_dir}")
            continue

        image_files = list_image_files(image_dir)
        mask_files = list_image_files(mask_dir)

        image_map = {p.name: p for p in image_files}
        mask_map = {p.name: p for p in mask_files}

        common_names = sorted(set(image_map.keys()) & set(mask_map.keys()))

        print(
            f"{channel_name} | {split_name} | "
            f"images={len(image_files)} masks={len(mask_files)} matched={len(common_names)}"
        )

        if len(common_names) == 0:
            print(f"[DEBUG] Sample image names: {[p.name for p in image_files[:5]]}")
            print(f"[DEBUG] Sample mask names : {[p.name for p in mask_files[:5]]}")

        for fname in common_names:
            rows.append(
                {
                    "channel": channel_name,
                    "split": split_name,
                    "filename": fname,
                    "image_src": str(image_map[fname]),
                    "mask_src": str(mask_map[fname]),
                }
            )

    return rows


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_dir(IMAGE_OUT)
    clean_dir(MASK_OUT)

    extracted = extract_all_zips()
    if not extracted:
        return

    all_rows = []

    for extracted_channel_dir in extracted:
        channel_root = resolve_channel_root(extracted_channel_dir)
        channel_name = channel_root.name.lower()

        print(f"\nScanning channel: {channel_name}")
        rows = collect_pairs_for_channel(channel_root, channel_name)
        all_rows.extend(rows)

    if not all_rows:
        print("No matched image-mask pairs found.")
        return

    df = pd.DataFrame(all_rows)
    print(f"\nTotal matched pairs before copying: {len(df)}")

    saved_rows = []
    used_names = set()

    for _, row in df.iterrows():
        original_name = row["filename"]
        new_name = f"{row['channel']}_{row['split']}_{original_name}"

        if new_name in used_names:
            stem = Path(original_name).stem
            suffix = Path(original_name).suffix
            i = 1
            while True:
                candidate = f"{row['channel']}_{row['split']}_{stem}_{i}{suffix}"
                if candidate not in used_names:
                    new_name = candidate
                    break
                i += 1

        used_names.add(new_name)

        image_dst = IMAGE_OUT / new_name
        mask_dst = MASK_OUT / new_name

        shutil.copy2(row["image_src"], image_dst)
        shutil.copy2(row["mask_src"], mask_dst)

        saved_rows.append(
            {
                "image_id": new_name,
                "channel": row["channel"],
                "split": row["split"],
                "original_filename": original_name,
                "image_path": str(image_dst),
                "mask_path": str(mask_dst),
            }
        )

    out_df = pd.DataFrame(saved_rows)
    out_df.to_csv(META_OUT, index=False)

    print("\nFinished preparing dataset.")
    print(f"Images saved to: {IMAGE_OUT}")
    print(f"Masks saved to : {MASK_OUT}")
    print(f"Metadata saved : {META_OUT}")

    print("\nCounts by channel:")
    print(out_df["channel"].value_counts())

    print("\nCounts by split:")
    print(out_df["split"].value_counts())


if __name__ == "__main__":
    main()
