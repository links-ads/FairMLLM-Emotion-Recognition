import matplotlib.pyplot as plt

def plot_fairness_vs_hparam(df, hparam, fairness_metrics, model, dataset, fold, show_std=True):
    """
    Plot F1 Macro and selected fairness metrics vs hyperparameter.
    
    Args:
        df: DataFrame with computed fairness metrics
        hparam: 'temperature' or 'top_p'
        fairness_metrics: list of fairness metric names to plot
        model: model name
        dataset: dataset name
        fold: fold number
    """
    agg_dict = {'global_f1_macro': ['mean', 'std']}
    for metric in fairness_metrics:
        agg_dict[metric] = ['mean', 'std']
    
    grouped = df.groupby(hparam).agg(agg_dict).reset_index()
    grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in grouped.columns]
    grouped = grouped.sort_values(hparam)
    
    colors = {'statistical_parity': '#A23B72', 'equal_opportunity': '#F18F01', 
              'equal_non_opportunity': '#C73E1E', 'predictive_parity': '#6A4C93',
              'negative_predictive_parity': '#1982C4'}
    markers = {'statistical_parity': 's', 'equal_opportunity': '^', 
               'equal_non_opportunity': 'v', 'predictive_parity': 'D',
               'negative_predictive_parity': 'X'}
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # F1 Macro (left y-axis)
    ax1.plot(grouped[hparam], grouped['global_f1_macro_mean'], marker='o', linewidth=3, 
             markersize=10, label='F1 Macro', color='#2E86AB')
    if show_std:
        ax1.fill_between(grouped[hparam], 
                        grouped['global_f1_macro_mean'] - grouped['global_f1_macro_std'], 
                        grouped['global_f1_macro_mean'] + grouped['global_f1_macro_std'], 
                        alpha=0.3, color='#2E86AB')
    ax1.set_xlabel(hparam.capitalize(), fontsize=16, fontweight='bold')
    ax1.set_ylabel('Global F1 Macro Score', fontsize=16, fontweight='bold', color='#2E86AB')
    ax1.tick_params(axis='both', labelsize=14, labelcolor='#2E86AB')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2 = ax1.twinx()
    y_max = 0
    
    for metric in fairness_metrics:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        label = metric.replace('_', ' ').title()
        
        ax2.plot(grouped[hparam], grouped[mean_col], marker=markers[metric], linewidth=3, 
                 markersize=10, label=label, color=colors[metric])
        if show_std:
            ax2.fill_between(grouped[hparam], 
                            grouped[mean_col] - grouped[std_col], 
                            grouped[mean_col] + grouped[std_col], 
                            alpha=0.3, color=colors[metric])
        y_max = max(y_max, grouped[mean_col].max())
    
    ax2.set_ylabel('Fairness Disparity', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_ylim(0, y_max * 1.2)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='best')
    
    fold = fold if fold is not None else 'All'
    plt.title(f'{model} on {dataset} Fold {fold}', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def plot_fairness_by_emotion(df, hparam, fairness_metric, model, dataset, fold, show_std=True):
    """
    Plot a specific fairness metric across emotions vs hyperparameter.
    
    Args:
        df: DataFrame with computed fairness metrics by emotion
        hparam: 'temperature' or 'top_p'
        fairness_metric: one of the fairness metric names (without emotion suffix)
        model: model name
        dataset: dataset name
        fold: fold number
    """
    emotions = ['Angry', 'Happy', 'Neutral', 'Sad']
    colors = {'Angry': '#E63946', 'Happy': '#06FFA5', 'Neutral': '#4A5859', 'Sad': '#457B9D'}
    markers = {'Angry': 'o', 'Happy': '^', 'Neutral': 's', 'Sad': 'v'}
    
    # Prepare aggregation dict
    agg_dict = {}
    for emotion in emotions:
        col_name = f'{fairness_metric}_{emotion}'
        agg_dict[col_name] = ['mean', 'std']
    
    # Group by hyperparameter
    grouped = df.groupby(hparam).agg(agg_dict).reset_index()
    grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in grouped.columns]
    grouped = grouped.sort_values(hparam)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    for emotion in emotions:
        mean_col = f'{fairness_metric}_{emotion}_mean'
        std_col = f'{fairness_metric}_{emotion}_std'
        
        ax.plot(grouped[hparam], grouped[mean_col], marker=markers[emotion], linewidth=3, 
                markersize=10, label=emotion, color=colors[emotion])
        if show_std:
            ax.fill_between(grouped[hparam], 
                            grouped[mean_col] - grouped[std_col], 
                            grouped[mean_col] + grouped[std_col], 
                            alpha=0.2, color=colors[emotion])

    fairness_metric = fairness_metric.replace('_', ' ').title() 
    ax.set_xlabel(hparam.capitalize(), fontsize=16, fontweight='bold')
    ax.set_ylabel(fairness_metric, fontsize=16, fontweight='bold', color='#2E86AB')
    ax.tick_params(axis='both', labelsize=14, labelcolor='#2E86AB')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', title='Emotion', title_fontsize=14)
    
    fold = fold if fold is not None else 'All'
    plt.title(f'{model} on {dataset} Fold {fold}', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()