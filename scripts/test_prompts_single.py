import os
import sys
import argparse
import torch
import pandas as pd

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
from sklearn.metrics import f1_score
from EmoBox.EmoBox import EmoDataset
from mllm_emotion_classifier.models import ModelFactory

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_prompt(model, test_dataset, num_samples=None):
    subset = torch.utils.data.Subset(test_dataset, range(num_samples)) if num_samples else test_dataset
    loader = torch.utils.data.DataLoader(subset, batch_size=4, num_workers=4, collate_fn=model.collate_fn)
    
    predictions, labels = [], []
    for inputs, lbl in tqdm(loader, desc=model.prompt_name, leave=False):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        preds = model.predict(inputs)
        predictions.extend(preds)
        labels.extend(lbl)
    
    valid = [(p, l) for p, l in zip(predictions, labels) if p is not None]
    if not valid:
        return None
    
    y_pred, y_true = zip(*valid)
    return {
        'prompt': model.prompt_name,
        'f1_macro': round(f1_score(y_true, y_pred, average='macro'), 4),
        'f1_weighted': round(f1_score(y_true, y_pred, average='weighted'), 4),
        'valid_rate': round(len(valid) / len(predictions), 4) #! labels
    }

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
    parser.add_argument("--output_dir", type=str, default="outputs/test_prompts/runs")
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
    
    metrics = test_prompt(model, test, num_samples=args.num_samples)
    
    if metrics:
        metrics['run'] = args.run_id
        metrics['model'] = args.model
        metrics['dataset'] = args.dataset
        metrics['fold'] = args.fold
        
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        output_file = outdir / f"run_{args.run_id}_{args.prompt}.csv"
        pd.DataFrame([metrics]).to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")
        print(f"F1 macro: {metrics['f1_macro']:.4f}")
    
    return metrics

if __name__ == "__main__":
    main()