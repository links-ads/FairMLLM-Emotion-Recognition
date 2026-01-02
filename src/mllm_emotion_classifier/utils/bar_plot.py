import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_global_over_attribute_values(df, top_p, attribute, attribute_values, temperature=False, metric='f1_macro', 
                                  model=None, dataset=None, fold=None, outpath=None):
    """
    Plot bar comparison of a single global metric between Male and Female.
    
    Args:
        df: DataFrame with gender-specific metrics
        metric: global metric name to compare ('f1_macro', 'f1_weighted', 'accuracy_unweighted', 'accuracy_weighted')
        model: model name for title
        dataset: dataset name for title
        fold: fold number for title
    """
    if temperature:
        df = df[df['temperature'] == top_p]
    else:
        df = df[df['top_p'] == top_p]
    
    assert metric in ['f1_macro', 'f1_weighted', 'accuracy_unweighted', 'accuracy_weighted'], "Invalid metric"
    
    attribute_values = sorted(list(attribute_values))
    
    # cmap = cm.get_cmap('tab10')
    # colors = {e: cmap(i % 10) for i, e in enumerate(attribute_values)}
    
    data = []
    for value in attribute_values:
        col_name = f'{attribute}_{value}_global_{metric}'
        if col_name in df.columns:
            data.append(df[col_name].mean() * 100)
        else:
            data.append(0)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    x = np.arange(len(attribute_values))
    width = 0.6
    
    bars = ax.bar(x, data, width, color='#3498db', alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    metric_label = metric.replace('_', '-').title()
    ax.set_ylabel(metric_label, fontsize=20, fontweight='bold')
    # ax.set_title(f'{dataset}', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x)
    # Replace spaces with newlines for multi-word labels
    formatted_labels = [label.replace(' ', '\n') for label in attribute_values]
    ax.set_xticklabels(formatted_labels, fontsize=17)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='both', labelsize=17)
    
    # if model and dataset:
    #     fold_str = fold if fold is not None else 'All'
    #     plt.suptitle(f'{model} on {dataset} Fold {fold_str}', fontsize=12, y=0.98)
    
    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()


def plot_classwise_over_attribute_values(df, top_p, emotions, attribute, attribute_values, temperature=False, metric='accuracy', 
                                      model=None, dataset=None, fold=None, outpath=None):
    """
    Plot bar comparison of emotion-specific metrics between Male and Female.
    
    Args:
        df: DataFrame with gender-specific metrics by emotion
        emotions: list of emotion names
        metric: one of 'accuracy', 'true_positive_rate', 'false_positive_rate', 'f1_score'
        model: model name for title
        dataset: dataset name for title
        fold: fold number for title
    """
    if temperature:
        df = df[df['temperature'] == top_p]
    else:
        df = df[df['top_p'] == top_p]

    assert metric in ['accuracy', 'true_positive_rate', 'false_positive_rate', 'f1_score'], "Invalid metric"

    color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    attribute_values = sorted(list(attribute_values))
    colors = {value: color_palette[i % len(color_palette)] for i, value in enumerate(attribute_values)}
    
    emotions_list = sorted(list(emotions))
    data = {value: [] for value in attribute_values}
    
    for emotion in emotions_list:
        for value in attribute_values:
            col_name = f'{attribute}_{value}_classwise_{metric}_{emotion}'
            if col_name in df.columns:
                data[value].append(df[col_name].mean() * 100)
            else:
                data[value].append(0)
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    x = np.arange(len(emotions_list))
    n_values = len(attribute_values)
    width = 0.8 / n_values
    
    offset = width * (n_values - 1) / 2
    
    for i, value in enumerate(attribute_values):
        bar_positions = x - offset + i * width
        # Format legend labels with newlines
        formatted_value = value.replace(' ', '\n')
        bars = ax.bar(bar_positions, data[value], width, label=formatted_value, 
                     color=colors[value], alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    metric_label = metric.replace('_', ' ').title()
    ax.set_ylabel(metric_label, fontsize=18, fontweight='bold')
    ax.set_title(f'{dataset}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    # Replace spaces with newlines for multi-word emotion labels
    formatted_emotions = [emotion.replace(' ', '\n') for emotion in emotions_list]
    ax.set_xticklabels(formatted_emotions, fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, title=attribute.title(), title_fontsize=14)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='both', labelsize=14)
    
    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()