from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

META_PATH = Path("data/processed/metadata.csv")
OUT_PATH = Path("data/folds/splits.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {META_PATH}")

    df = pd.read_csv(META_PATH)

    print("Total samples:", len(df))
    print("\nCounts by split:")
    print(df["split"].value_counts())

    print("\nCounts by channel:")
    print(df["channel"].value_counts())

    trainval_df = df[df["split"] == "trainval"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print("\nTrainval samples:", len(trainval_df))
    print("Test samples    :", len(test_df))

    trainval_df["fold"] = -1
    test_df["fold"] = -1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # We stratify by channel so each fold gets a similar green/red/yellow mix.
    for fold, (_, val_idx) in enumerate(skf.split(trainval_df, trainval_df["channel"])):
        trainval_df.loc[val_idx, "fold"] = fold

    final_df = pd.concat([trainval_df, test_df], ignore_index=True)
    final_df = final_df.sort_values(["split", "channel", "image_id"]).reset_index(drop=True)

    final_df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved splits to: {OUT_PATH}")

    print("\nTrainval fold counts:")
    print(trainval_df["fold"].value_counts().sort_index())

    print("\nFold x channel table:")
    print(pd.crosstab(trainval_df["fold"], trainval_df["channel"]))

    print("\nTest samples remain holdout with fold = -1")
    print(test_df[["image_id", "channel", "split", "fold"]].head())


if __name__ == "__main__":
    main()
