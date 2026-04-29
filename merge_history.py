from pathlib import Path
import glob
import pandas as pd


OUT_DIR = Path("outputs/metrics")


def main():
    history_files = glob.glob(str(OUT_DIR / "history_*.csv"))

    if not history_files:
        print("No history files found.")
        return

    all_histories = []
    for file_path in history_files:
        df = pd.read_csv(file_path)
        df["source_file"] = Path(file_path).name
        all_histories.append(df)

    history_all = pd.concat(all_histories, ignore_index=True)
    history_all = history_all.sort_values(["model", "encoder", "fold", "epoch"]).reset_index(drop=True)

    history_all_path = OUT_DIR / "history_all_epochs.csv"
    history_all.to_csv(history_all_path, index=False)

    # Best epoch per model+encoder+fold by validation Dice
    idx = history_all.groupby(["model", "encoder", "fold"])["dice"].idxmax()
    best_per_fold = history_all.loc[idx].copy()
    best_per_fold = best_per_fold.sort_values(["model", "encoder", "fold"]).reset_index(drop=True)

    best_per_fold_path = OUT_DIR / "history_best_per_fold.csv"
    best_per_fold.to_csv(best_per_fold_path, index=False)

    # Fold-aggregated stats based on each fold's best epoch
    best_summary = best_per_fold.groupby(["model", "encoder"]).agg({
        "dice": ["mean", "std"],
        "iou": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "epoch": ["mean"],
    }).reset_index()

    best_summary.columns = [
        "model", "encoder",
        "dice_mean", "dice_std",
        "iou_mean", "iou_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "best_epoch_mean",
    ]
    best_summary = best_summary.sort_values("dice_mean", ascending=False).reset_index(drop=True)

    best_summary_path = OUT_DIR / "history_best_summary.csv"
    best_summary.to_csv(best_summary_path, index=False)

    print(f"Saved: {history_all_path}")
    print(f"Saved: {best_per_fold_path}")
    print(f"Saved: {best_summary_path}")
    print("\nTop models by mean best-fold Dice:")
    print(best_summary[["model", "encoder", "dice_mean", "dice_std", "best_epoch_mean"]])


if __name__ == "__main__":
    main()
