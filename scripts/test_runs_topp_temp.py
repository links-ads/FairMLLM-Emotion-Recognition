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
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--model", type=str, default="qwen2-audio-instruct")
    parser.add_argument("--temperature", type=float, default=-1)
    parser.add_argument("--top_p", type=float, default=-1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--prompt", nargs='+', type=str, default=["simple", "task"])
    parser.add_argument("--output_dir", type=str, default="outputs/")
    return parser.parse_args()

def run_single(args, fold, run_id, prompt):
    cmd = ["python", "scripts/utils/test_single_run_topp_temp.py", 
        "--dataset", args.dataset,
        "--data_dir", args.data_dir,
        "--meta_data_dir", args.meta_data_dir,
        "--fold", str(fold),
        "--model", args.model,
        "--prompt", prompt,
        "--temperature", str(args.temperature),
        "--language", str(args.language),
        "--top_p", str(args.top_p),
        "--run_id", str(run_id),
        "--output_dir", "outputs/"]
    
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

def main():
    args = parse_args()

    dataset_path = Path(args.meta_data_dir) / args.dataset
    n_folds = len([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    if float(args.temperature) != -1.0:
        print(f"Running {n_folds} folds × {args.runs} runs × {len(args.prompt)} prompts = {n_folds * args.runs * len(args.prompt)} temperature = {args.temperature} experiments")
    if float(args.top_p) != -1.0:
        print(f"Running {n_folds} folds × {args.runs} runs × {len(args.prompt)} prompts = {n_folds * args.runs * len(args.prompt)} top_p = {args.top_p} experiments")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    if args.fold > 0:
        print(f"\n=== Fold {args.fold}/{n_folds} ===")
        for run_id in range(args.runs):
            for prompt in args.prompt:
                if not run_single(args, args.fold, run_id, prompt):
                    raise RuntimeError(f"Run {run_id} Fold {args.fold} Prompt {prompt} failed.")
    else:
        for fold in range(1, n_folds + 1):
            print(f"\n=== Fold {fold}/{n_folds} ===")
            for run_id in range(args.runs):
                for prompt in args.prompt:
                    if not run_single(args, fold, run_id, prompt):
                        raise RuntimeError(f"Run {run_id} Fold {fold} Prompt {prompt} failed.")

if __name__ == "__main__":
    main()