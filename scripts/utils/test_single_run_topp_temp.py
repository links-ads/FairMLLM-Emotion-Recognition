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
    metrics = flatten_dict(results['metrics'])
    return {
        'valid_rate': results['valid_rate'],
        **metrics
    }

def save_csv(args, metrics, output_file):
    stats_df = pd.DataFrame([metrics])
    
    if output_file.exists():
        existing_df = pd.read_csv(output_file)
        mask = (
            (existing_df['run'] == metrics['run']) & 
            (existing_df['dataset'] == metrics['dataset']) & 
            (existing_df['fold'] == metrics['fold']) &
            (existing_df['model'] == metrics['model']) & 
            (existing_df['prompt'] == metrics['prompt']) 
        )
        if float(args.temperature) != -1.0:
            mask &= (existing_df['temperature'] == metrics['temperature'])
        if float(args.top_p) != -1.0:
            mask &= (existing_df['top_p'] == metrics['top_p'])
        existing_df = existing_df[~mask]
        stats_df = pd.concat([existing_df, stats_df], ignore_index=True)
    
    if float(args.temperature) != -1.0:
        stats_df = stats_df.sort_values(['run', 'dataset', 'fold', 'model', 'prompt', 'temperature']).reset_index(drop=True)
    if float(args.top_p) != -1.0:
        stats_df = stats_df.sort_values(['run', 'dataset', 'fold', 'model', 'prompt', 'top_p']).reset_index(drop=True)

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
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=-1)
    parser.add_argument("--top_p", type=float, default=-1)
    parser.add_argument("--output_dir", type=str, default="outputs/")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Run {args.run_id} - Prompt: {args.prompt}")
    
    test = EmoDataset(args.dataset, args.data_dir, args.meta_data_dir, fold=args.fold, split="test", language=args.language)
    
    model = ModelFactory.create(
        name=args.model,
        class_labels=set(test.label_map.values()),
        prompt_name=args.prompt,
        temperature=float(args.temperature) if float(args.temperature) != -1.0 else 1.0,
        top_p=float(args.top_p) if float(args.top_p) != -1.0 else 1.0,
        device=device,
    )
    
    metrics = test_run(model, test, num_samples=args.num_samples)
    
    if metrics:
        metadata = {
            'run': args.run_id,
            'dataset': args.dataset,
            'fold': args.fold,
            'model': args.model,
            'prompt': args.prompt,
        }
        
        if float(args.temperature) != -1.0:
            metadata['temperature'] = float(args.temperature)
            outdir = Path(args.output_dir) / "temperature_runs"
        if float(args.top_p) != -1.0:
            metadata['top_p'] = float(args.top_p)
            outdir = Path(args.output_dir) / "topp_runs"

        metrics = {**metadata, **metrics}
        
        outdir = outdir / f"{args.model}" / f"{args.dataset}"
        outdir.mkdir(parents=True, exist_ok=True)
        output_file = outdir / f"fold_{args.fold}.csv"
        save_csv(args, metrics, output_file)
    
    return metrics

if __name__ == "__main__":
    main()