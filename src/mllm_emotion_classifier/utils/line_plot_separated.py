import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_fairness_vs_hparam(
        df, hparam, fairness_metrics, sensitive_attrs, model, dataset, fold, show_std=True, output_path=None
    ):
    """
    Plot F1 Macro and selected fairness metrics vs hyperparameter in separate subplots.
    
    Args:
        df: DataFrame with computed fairness metrics
        hparam: 'temperature' or 'top_p'
        fairness_metrics: list of fairness metric names to plot (without sensitive_attr prefix)
        sensitive_attrs: list of sensitive attribute names (e.g., ['gender', 'language'])
        model: model name
        dataset: dataset name
        fold: fold number
        show_std: whether to show standard deviation bands
        output_path: path to save the figure (optional)
    """
    agg_dict = {'global_f1_macro': ['mean', 'std']}
    
    # Add all fairness metrics for all sensitive attributes
    for sens_attr in sensitive_attrs:
        for metric in fairness_metrics:
            col_name = f'{sens_attr}_{metric}'
            agg_dict[col_name] = ['mean', 'std']
    
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
    
    # Create figure with different row heights
    # F1 plot gets height ratio 1.5, fairness plots get height ratio 1
    n_rows = 1 + len(sensitive_attrs)  # 1 for F1, rest for fairness
    height_ratios = [1.5] + [1] * len(sensitive_attrs)
    
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3 + 2.5 * len(sensitive_attrs)), 
                             gridspec_kw={'height_ratios': height_ratios})
    
    # If only one sensitive attribute, axes is not a list
    if n_rows == 2:
        axes = [axes[0], axes[1]]
    
    # Collect all fairness handles and labels for combined legend
    fairness_handles = []
    fairness_labels = []
    
    # ===== TOP PLOT: F1 Macro =====
    ax1 = axes[0]
    f1_line, = ax1.plot(grouped[hparam], grouped['global_f1_macro_mean'], marker='o', linewidth=4, 
             markersize=12, label='F1 Macro', color='#2E86AB')
    if show_std:
        ax1.fill_between(grouped[hparam], 
                        grouped['global_f1_macro_mean'] - grouped['global_f1_macro_std'], 
                        grouped['global_f1_macro_mean'] + grouped['global_f1_macro_std'], 
                        alpha=0.3, color='#2E86AB')
    
    # Dynamic y-axis for F1 Macro
    if show_std:
        f1_min = (grouped['global_f1_macro_mean'] - grouped['global_f1_macro_std']).min()
        f1_max = (grouped['global_f1_macro_mean'] + grouped['global_f1_macro_std']).max()
    else:
        f1_min = grouped['global_f1_macro_mean'].min()
        f1_max = grouped['global_f1_macro_mean'].min()
    
    ax1.set_ylim(max(0, f1_min - 0.03), min(1.0, f1_max + 0.03))
    ax1.set_ylabel('Score', fontsize=24, fontweight='bold')
    ax1.set_title('F1 Macro', fontsize=26, fontweight='bold', pad=15)
    ax1.set_xticks(grouped[hparam])
    ax1.set_xticklabels([])  # Hide x-tick labels
    ax1.tick_params(axis='both', labelsize=22)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ===== FAIRNESS PLOTS: One row per sensitive attribute =====
    for idx, sens_attr in enumerate(sensitive_attrs):
        ax = axes[idx + 1]
        all_values = []
        
        for metric in fairness_metrics:
            mean_col = f'{sens_attr}_{metric}_mean'
            std_col = f'{sens_attr}_{metric}_std'
            
            # Skip if column doesn't exist
            if mean_col not in grouped.columns:
                continue
            
            base_metric = metric.replace(f'{sens_attr}_', '').replace(f'{sens_attr.lower()}_', '')
            label = base_metric.replace('_', ' ').title()
            
            color = colors.get(base_metric, '#333333')
            marker = markers.get(base_metric, 'o')
            
            line, = ax.plot(grouped[hparam], grouped[mean_col], marker=marker, linewidth=4, 
                     markersize=12, label=label, color=color)
            
            # Collect handles and labels for combined legend (only from first sens_attr)
            if idx == 0:
                fairness_handles.append(line)
                fairness_labels.append(label)
            
            if show_std:
                ax.fill_between(grouped[hparam], 
                                grouped[mean_col] - grouped[std_col], 
                                grouped[mean_col] + grouped[std_col], 
                                alpha=0.3, color=color)
                all_values.extend((grouped[mean_col] - grouped[std_col]).tolist())
                all_values.extend((grouped[mean_col] + grouped[std_col]).tolist())
            else:
                all_values.extend(grouped[mean_col].tolist())
        
        # Dynamic y-axis for Fairness Metrics
        if all_values:
            fairness_min = min(all_values)
            fairness_max = max(all_values)
            ax.set_ylim(max(0, fairness_min - 0.03), min(1.0, fairness_max + 0.03))
        
        # Only show x-label on last plot
        if idx == len(sensitive_attrs) - 1:
            ax.set_xlabel(hparam.capitalize(), fontsize=24, fontweight='bold')
        else:
            ax.set_xticklabels([])  # Hide x-tick labels for intermediate plots
        
        # Set title instead of ylabel
        ax.set_ylabel('Unfairness', fontsize=24, fontweight='bold')
        sens_attr = 'age' if 'age' in sens_attr else sens_attr
        ax.set_title(f'{sens_attr.title()}', fontsize=26, fontweight='bold', pad=15)
        ax.set_xticks(grouped[hparam])
        ax.tick_params(axis='both', labelsize=22)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add combined legend to F1 plot with bigger font
    all_handles = [f1_line] + fairness_handles
    all_labels = ['F1 Macro'] + fairness_labels
    ax1.legend(all_handles, all_labels, fontsize=16, loc='best', ncol=2)
    
    plt.tight_layout()
    plt.show()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')