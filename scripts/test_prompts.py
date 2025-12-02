import argparse
import time
import shutil
import subprocess
import pandas as pd

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iemocap")
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--meta_data_dir", type=str, default="EmoBox/data/")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--model", type=str, default="qwen2-audio-instruct")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prompt", nargs='+', type=str, default=["simple", "task"])
    parser.add_argument("--output_dir", type=str, default="outputs/test_prompts")
    return parser.parse_args()

def run_single(args, fold, run_id, prompt):
    cmd = ["python", "scripts/test_prompts_single.py", 
        "--dataset", args.dataset,
        "--data_dir", args.data_dir,
        "--meta_data_dir", args.meta_data_dir,
        "--fold", str(fold),
        "--model", args.model,
        "--prompt", prompt,
        "--run_id", str(run_id),
        "--output_dir", f"{args.output_dir}/runs"]
    
    if args.num_samples:
        cmd.extend(["--num_samples", str(args.num_samples)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n❌ Run {run_id}/{prompt} FAILED")
        print(f"STDERR:\n{result.stderr}")
        print(f"STDOUT:\n{result.stdout}")
        return False
    
    print(f"Run {run_id}/{prompt}: ✓")
    return True

def aggregate(args, fold):
    csv_files = list((Path(args.output_dir) / "runs").glob("run_*.csv"))
    if not csv_files:
        return None, None
    
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    summary = df.groupby('prompt').agg({'f1_macro': ['mean', 'std'], 'f1_weighted': ['mean', 'std'],
                                         'valid_rate': ['mean', 'std']}).round(4)
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()
    for i, (k, v) in enumerate(
        [('model', args.model), ('dataset', args.dataset), ('fold', fold),('num_runs', args.runs)]
    ):
        summary.insert(i, k, v)
    return df, summary

def main():
    args = parse_args()

    dataset_path = Path(args.meta_data_dir) / args.dataset
    n_folds = len([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    print(f"Running {n_folds} folds × {args.runs} runs × {len(args.prompt)} prompts = {n_folds * args.runs * len(args.prompt)} experiments")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_file = outdir / "all_runs.csv"
    
    for fold in range(1, n_folds + 1):
        print(f"\n=== Fold {fold}/{n_folds} ===")
        for run_id in range(args.runs):
            for prompt in args.prompt:
                if not run_single(args, fold, run_id, prompt):
                    return
    
        df, summary = aggregate(args, fold)
        print(f"\n{summary.to_string(index=False)}")
        print(f"\nBest: {summary.loc[summary['f1_macro_mean'].idxmax(), 'prompt']}")

        if output_file.exists():
            summary.to_csv(output_file, mode='a', header=False, index=False)
        else:
            summary.to_csv(output_file, mode='w', header=True, index=False)

        runs_dir = Path(args.output_dir) / "runs"
        shutil.rmtree(runs_dir, ignore_errors=True)
        time.sleep(2)

if __name__ == "__main__":
    main()