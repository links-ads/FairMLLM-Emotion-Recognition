import os
from pathlib import Path

import pandas as pd
from mllm_emotion_classifier.utils import add_fairness_metrics_to_df
from EmoBox.EmoBox import EmoDataset

sensitive_attr_dict = {
    'iemocap': ['gender'],
    'cremad': ['gender', 'age', 'ethnicity', 'race'],
    'emovdb': ['gender'],
    'ravdess': ['gender'],
    'meld': ['gender'],
}

models = [
    'qwen2-audio-instruct',
    'audio-flamingo-3',
    'voxtral-mini',
    'salmonn-7b',
]

hparam = 'temperature'  # or 'top_p'
assert hparam in ['temperature', 'top_p'], "hparam must be either 'temperature' or 'top_p'"

datasets = ['iemocap', 'meld', 'cremad', 'ravdess', 'emovdb']
metadata_dir = Path('EmoBox/data/')
out_dir = Path('outputs-3') / "temperature_runs" if hparam == 'temperature' else Path('outputs-2') / "topp_runs"

results = []

for dataset in datasets:
    sensitive_attrs = sensitive_attr_dict[dataset]
    dataset_path = metadata_dir / dataset
    n_folds = len([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("fold_")])

    for model in models:
        if not (out_dir / model / dataset).exists():
            print(f"Skipping {model} on {dataset} (no results found)")
            continue

        dfs = []
        for f in range(1, n_folds + 1):
            results_csv = out_dir / model / dataset / f'fold_{f}.csv'
            if not results_csv.exists():
                print(f"Skipping fold {f} for {model} on {dataset} (file not found)")
                continue
            df_fold = pd.read_csv(results_csv)
            dfs.append(df_fold)

        if not dfs:
            print(f"No data for {model} on {dataset}")
            continue

        df = pd.concat(dfs, ignore_index=True)

        cols = [hparam, 'global_f1_macro', 'global_accuracy_unweighted'] + \
               [f"{attr}_{metric}" for attr in sensitive_attrs
                for metric in ['statistical_parity', 'equal_opportunity', 'overall_accuracy_equality']]

        grouped_stats = df[cols].groupby([hparam]).agg(['mean', 'std']).reset_index()

        grouped = grouped_stats[[hparam]].copy()
        for col in cols:
            if col == hparam:
                continue
            mean_vals = (grouped_stats[(col, 'mean')] * 100).round(2)
            std_vals = (grouped_stats[(col, 'std')] * 100).round(2)
            grouped[col] = mean_vals.astype(str) + ' ± ' + std_vals.astype(str)

        best_idx = (grouped_stats[('global_f1_macro', 'mean')]).idxmax()
        best_row = grouped.loc[best_idx]
        best_row_str = "\n".join([
            f"{col[0]}: {best_row[col]}" if isinstance(col, tuple) else f"{col}: {best_row[col]}"
            for col in grouped.columns
        ])
        results.append(f"{dataset.upper()} - {model}\n{best_row_str}\n\n")

output_file = Path('best_results_no_postprocessing.txt')
with open(output_file, 'w') as f:
    f.writelines(results)

print(f"Results saved to {output_file}")