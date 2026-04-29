from pathlib import Path
import glob
import pandas as pd

OUT_DIR = Path("outputs/metrics")


def main():
    files = glob.glob(str(OUT_DIR / "result_*.csv"))

    if not files:
        print("No result files found.")
        return

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUT_DIR / "all_results.csv", index=False)

    summary = df.groupby(["model", "encoder", "split"]).agg({
        "dice": ["mean", "std"],
        "iou": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
    }).reset_index()

    summary.columns = [
        "model", "encoder", "split",
        "dice_mean", "dice_std",
        "iou_mean", "iou_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
    ]

    summary = summary.sort_values(["split", "dice_mean"], ascending=[True, False])
    summary.to_csv(OUT_DIR / "summary_results.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
