import numpy as np

def quantile_binning(attr_values: np.ndarray, n_bins: int = 4):
    quantiles = np.percentile(attr_values, [q * 100 / n_bins for q in range(1, n_bins)])
    bins_quantile = [attr_values.min()] + list(quantiles) + [attr_values.max()]
    binned_quantile = np.digitize(attr_values, bins_quantile[:-1]) - 1
    labels_quantile = []
    for i in range(n_bins):
        if i == 0:
            labels_quantile.append(f'{bins_quantile[0]:.0f}-{quantiles[0]:.0f}')
        elif i == n_bins - 1:
            labels_quantile.append(f'{quantiles[-1]:.0f}-{bins_quantile[-1]:.0f}')
        else:
            labels_quantile.append(f'{quantiles[i-1]:.0f}-{quantiles[i]:.0f}')
    
    for i, label in enumerate(labels_quantile):
        count = np.sum(binned_quantile == i)
        print(f"  {label}: {count} ({count/len(attr_values)*100:.1f}%)")

    binned_labels = np.array([labels_quantile[i] for i in binned_quantile])
    return binned_labels
