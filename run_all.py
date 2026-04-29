import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = ["Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN"]


def run_command(cmd):
    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def parse_models(models_arg):
    if not models_arg:
        return DEFAULT_MODELS
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def parse_folds(folds_arg):
    return [int(x.strip()) for x in folds_arg.split(",") if x.strip()]


def main(args):
    project_root = Path(__file__).resolve().parent.parent
    python_exe = sys.executable
    metrics_dir = project_root / "outputs" / "metrics"

    models = parse_models(args.models)
    folds = parse_folds(args.folds)

    print("Project root:", project_root)
    print("Python      :", python_exe)
    print("Models      :", models)
    print("Folds       :", folds)
    print("Epochs      :", args.epochs)
    print("Batch size  :", args.batch_size)
    print("Image size  :", args.img_size)
    print("LR          :", args.lr)
    print("Patience    :", args.patience)
    print("Split eval  :", args.eval_split)

    for model_name in models:
        for fold in folds:
            result_path = metrics_dir / f"result_{model_name}_{args.encoder}_{args.eval_split}_fold{fold}.csv"
            if args.skip_existing and result_path.exists():
                print(f"Skipping completed run: {result_path.name}")
                continue

            train_cmd = [
                python_exe, "src/train.py",
                "--model", model_name,
                "--encoder", args.encoder,
                "--fold", str(fold),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--img_size", str(args.img_size),
                "--num_workers", str(args.num_workers),
                "--patience", str(args.patience),
            ]
            run_command(train_cmd)

            infer_cmd = [
                python_exe, "src/infer.py",
                "--model", model_name,
                "--encoder", args.encoder,
                "--fold", str(fold),
                "--split", args.eval_split,
                "--img_size", str(args.img_size),
                "--threshold", str(args.threshold),
            ]
            run_command(infer_cmd)

    summary_script = project_root / "src" / "summarize_results.py"
    legacy_summary_script = project_root / "src" / "summarise_result.py"

    if summary_script.exists():
        run_command([python_exe, str(summary_script)])
    elif legacy_summary_script.exists():
        run_command([python_exe, str(legacy_summary_script)])
    else:
        print("No summary script found; skipping summary generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    main(args)
