import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from EmoBox.EmoBox import EmoDataset
from mllm_emotion_classifier.evaluate import Statistics
from mllm_emotion_classifier.utils import flatten_dict

def collect_dataset_info(dataset):
    y_true = []
    sens_attr_dict = {k: [] for k in dataset[0].keys() if k not in {'audio', 'label', 'key', 'text'}}
    
    for sample in dataset:
        y_true.append(sample['label'])
        for k in sens_attr_dict.keys():
            sens_attr_dict[k].append(sample[k])
    
    y_true = np.array(y_true)
    sens_attr_dict = {k: np.array(v) for k, v in sens_attr_dict.items()}
    return y_true, sens_attr_dict

def compute_statistics(dataset_name, data_dir, meta_data_dir, fold):
    """Compute statistics for a dataset fold"""
    dataset = EmoDataset(dataset_name, data_dir, meta_data_dir, fold=fold, split="test")
    y_true, sens_attr_dict = collect_dataset_info(dataset)
    stats = Statistics(y_true, sens_attr_dict).compute()
    return {
        'dataset': dataset_name,
        'fold': fold,
        **flatten_dict(stats)
    }

def save_statistics(stats, output_file):
    """Save statistics to CSV, avoiding duplicates"""
    stats_df = pd.DataFrame([stats])
    
    if output_file.exists():
        existing_df = pd.read_csv(output_file)
        mask = (existing_df['dataset'] == stats['dataset']) & (existing_df['fold'] == stats['fold'])
        existing_df = existing_df[~mask]
        stats_df = pd.concat([existing_df, stats_df], ignore_index=True)
    
    stats_df = stats_df.sort_values(['dataset', 'fold']).reset_index(drop=True)
    stats_df.to_csv(output_file, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="./", help="Data directory")
    parser.add_argument("--meta_data_dir", type=str, default="EmoBox/data/", help="Metadata directory")
    parser.add_argument("--output_file", type=str, default="outputs/statistics/statistics.csv", help="Output CSV file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    dataset_path = Path(args.meta_data_dir) / args.dataset
    n_folds = len([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    for fold in range(1, n_folds + 1):
        print(f"\nProcessing {args.dataset} - Fold {fold}")
        print("=" * 60)
        
        try:
            stats = compute_statistics(args.dataset, args.data_dir, args.meta_data_dir, fold)
            save_statistics(stats, output_file)
        except Exception as e:
            print(f"Error processing fold {fold}: {e}")
            continue

if __name__ == "__main__":
    main()