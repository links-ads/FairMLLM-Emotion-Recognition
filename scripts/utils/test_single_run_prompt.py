import os
import sys
import argparse
import torch
import pandas as pd

from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from EmoBox.EmoBox import EmoDataset
from mllm_emotion_classifier.models import ModelFactory
from mllm_emotion_classifier.evaluate import Evaluator
from mllm_emotion_classifier.utils import flatten_dict

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_run(model, test_dataset, num_samples=None):
    subset = torch.utils.data.Subset(test_dataset, range(num_samples)) if num_samples else test_dataset
    loader = torch.utils.data.DataLoader(subset, batch_size=4, num_workers=4, collate_fn=model.collate_fn)

    evaluator = Evaluator()
    evaluator.evaluate(model, loader, n_samples=num_samples)
    
    results = evaluator.results
    # stats = flatten_dict(results['stats'])
    metrics = flatten_dict(results['metrics'])
    return {
        # 'stats': stats,
        'metrics': {
        'valid_rate': results['valid_rate'],
        **metrics
        }
    }


def save_statistics(stats, output_file):
    """Save statistics to CSV, avoiding duplicates"""
    stats_df = pd.DataFrame([stats])
    
    if output_file.exists():
        existing_df = pd.read_csv(output_file)
        mask = (
            (existing_df['run'] == stats['run']) & 
            (existing_df['prompt'] == stats['prompt']) & 
            (existing_df['model'] == stats['model']) & 
            (existing_df['dataset'] == stats['dataset']) & 
            (existing_df['fold'] == stats['fold'])
        )
        existing_df = existing_df[~mask]
        stats_df = pd.concat([existing_df, stats_df], ignore_index=True)
    
    stats_df = stats_df.sort_values(['dataset', 'fold', 'model', 'prompt', 'run']).reset_index(drop=True)
    stats_df.to_csv(output_file, index=False)
    print(f"Stats saved to: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--dataset", type=str, default="iemocap")
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--meta_data_dir", type=str, default="EmoBox/data/")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--model", type=str, default="qwen2-audio-instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/test_runs/runs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Run {args.run_id} - Prompt: {args.prompt}")
    
    test = EmoDataset(args.dataset, args.data_dir, args.meta_data_dir, fold=args.fold, split="test")
    
    model = ModelFactory.create(
        name=args.model,
        class_labels=set(test.label_map.values()),
        prompt_name=args.prompt,
        device=device,
    )
    
    results = test_run(model, test, num_samples=args.num_samples)
    
    if results:
        metadata = {
            'run': args.run_id,
            'prompt': args.prompt,
            'model': args.model,
            'dataset': args.dataset,
            'fold': args.fold
        }
        
        # stats = {**metadata, **results['stats']}
        metrics = {**metadata, **results['metrics']}
        
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        output_file = outdir / f"run_{args.run_id}_{args.prompt}.csv"
        pd.DataFrame([metrics]).to_csv(output_file, index=False)
        print(f"Metrics Saved to: {output_file}")

        # stats_dir = outdir.parent / "statistics"
        # stats_dir.mkdir(parents=True, exist_ok=True)
        # stats_file = stats_dir / f"{args.dataset}.csv"
        # save_statistics(stats, stats_file)
    
    return metrics

if __name__ == "__main__":
    main()