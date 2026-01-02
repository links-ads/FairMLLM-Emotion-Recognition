import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_classwise_heatmap(df, top_p, emotions, attribute, attribute_values, temperature=False, metric='accuracy', 
                           model=None, dataset=None, fold=None, outpath=None):
    """
    Plot heatmap of emotion-specific metrics across attribute values.
    
    Args:
        df: DataFrame with gender-specific metrics by emotion
        emotions: list of emotion names
        metric: one of 'accuracy', 'true_positive_rate', 'false_positive_rate', 'f1_score'
        model: model name for title
        dataset: dataset name for title
        fold: fold number for title
        outpath: path to save the figure
    """
    if temperature:
        df = df[df['temperature'] == top_p]
    else:
        df = df[df['top_p'] == top_p]

    assert metric in ['accuracy', 'true_positive_rate', 'false_positive_rate', 'f1_score'], "Invalid metric"
    
    attribute_values = sorted(list(attribute_values))
    emotions_list = sorted(list(emotions))
    
    # Create matrix: rows = emotions, columns = attribute values
    matrix = []
    for emotion in emotions_list:
        row = []
        for value in attribute_values:
            col_name = f'{attribute}_{value}_classwise_{metric}_{emotion}'
            if col_name in df.columns:
                row.append(df[col_name].mean() * 100)
            else:
                row.append(0)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Create heatmap with better colormap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add colorbar with better styling
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(metric.replace('_', ' ').title(), fontsize=20, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=17, width=2, length=6)
    
    # Add grid lines between cells
    ax.set_xticks(np.arange(len(attribute_values)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(emotions_list)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(attribute_values)))
    ax.set_yticks(np.arange(len(emotions_list)))
    
    formatted_values = [value.replace(' ', '\n') for value in attribute_values]
    formatted_emotions = [emotion.replace(' ', '\n') for emotion in emotions_list]
    
    ax.set_xticklabels(formatted_values, fontsize=17, fontweight='bold', rotation=45)
    ax.set_yticklabels(formatted_emotions, fontsize=17, fontweight='bold')
    
    # Position x-axis labels at top
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    
    # Add value annotations with adaptive text color
    for i in range(len(emotions_list)):
        for j in range(len(attribute_values)):
            val = matrix[i, j]
            # Use white text on dark backgrounds, black on light backgrounds
            text_color = 'white' if val < 50 else 'black'
            text = ax.text(j, i, f'{val:.1f}',
                          ha='center', va='center', color=text_color, 
                          fontsize=17, fontweight='bold')
    
    # Remove tick marks
    ax.tick_params(which='both', length=0)
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(2)
    
    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()