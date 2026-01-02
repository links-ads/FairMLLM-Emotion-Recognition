import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_fairness_vs_hparam(
        df, hparam, fairness_metrics, sensitive_attr, model, dataset, fold, show_std=True, output_path=None
    ):
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
              'negative_predictive_parity': '#1982C4', 'overall_accuracy_equality': '#8AC926',
              'global_f1_macro': '#2E86AB'}
    markers = {'statistical_parity': 's', 'equal_opportunity': '^', 
               'equal_non_opportunity': 'v', 'predictive_parity': 'D',
               'negative_predictive_parity': 'X', 'overall_accuracy_equality': 'P',
               'global_f1_macro': 'o'}
    
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    
    # Plot F1 Macro
    ax.plot(grouped[hparam], grouped['global_f1_macro_mean'], marker='o', linewidth=3, 
             markersize=10, label='F1 Macro', color='#2E86AB')
    if show_std:
        ax.fill_between(grouped[hparam], 
                        grouped['global_f1_macro_mean'] - grouped['global_f1_macro_std'], 
                        grouped['global_f1_macro_mean'] + grouped['global_f1_macro_std'], 
                        alpha=0.3, color='#2E86AB')

    # Plot fairness metrics
    for metric in fairness_metrics:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        label = metric.replace('_', ' ').title()
        
        ax.plot(grouped[hparam], grouped[mean_col], marker=markers[metric], linewidth=3, 
                 markersize=10, label=label, color=colors[metric])
        if show_std:
            ax.fill_between(grouped[hparam], 
                            grouped[mean_col] - grouped[std_col], 
                            grouped[mean_col] + grouped[std_col], 
                            alpha=0.3, color=colors[metric])
    
    ax.set_xlabel(hparam.capitalize(), fontsize=18, fontweight='bold')
    ax.set_ylabel('Score', fontsize=18, fontweight='bold')
    ax.set_xticks(grouped[hparam])
    ax.tick_params(axis='both', labelsize=18)
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best')
    
    # plt.title(f'{sensitive_attr.upper()} -- {dataset.upper()}', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=300)

def plot_fairness_by_emotion(df, emotions, hparam, fairness_metric, model, dataset, fold, show_std=True):
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
    cmap = cm.get_cmap('tab10')
    colors = {e: cmap(i % 10) for i, e in enumerate(emotions)}
    
    marker_list = ['o', '^', 's', 'v', 'D', 'p', '*', 'h', 'X', 'P']
    markers = {e: marker_list[i % len(marker_list)] for i, e in enumerate(emotions)}
    
    agg_dict = {}
    for emotion in emotions:
        col_name = f'{fairness_metric}_{emotion}'
        agg_dict[col_name] = ['mean', 'std']
    
    grouped = df.groupby(hparam).agg(agg_dict).reset_index()
    grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in grouped.columns]
    grouped = grouped.sort_values(hparam)
    
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
    ax.set_ylabel(fairness_metric, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', title='Emotion', title_fontsize=14)
    
    fold = fold if fold is not None else 'All'
    plt.title(f'{model} on {dataset} Fold {fold}', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()