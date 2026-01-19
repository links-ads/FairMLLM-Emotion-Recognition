import re
import numpy as np

def compute_fair_diff_row(
        row,
        emotions,
        sensitive_attr='gender',
        metric='classwise_accuracy'
    ):
    assert metric in ['classwise_accuracy', # Statistical Parity
                      'classwise_true_positive_rate', # Equal Opportunity
                      'classwise_false_positive_rate',  # Equal Non-Opportunity
                      'classwise_positive_predictive_value', # Predictive Parity
                      'classwise_negative_predictive_value',
                      'global_accuracy_weighted'], "Invalid metric" # Negative Predictive Parity

    if metric == 'global_accuracy_weighted':
        disparities = None
        pattern = f'{sensitive_attr}_([^_]+)_{metric}$'
        matching_cols = [col for col in row.index if re.match(pattern, col)]
        if len(matching_cols) >= 2:
            values = [row[col] for col in matching_cols]
            all_disparities = [abs(values[i] - values[j]) 
                              for i in range(len(values)) 
                              for j in range(i+1, len(values))]
            mean_disparity = sum(all_disparities) / len(all_disparities)
            disparities = mean_disparity
    else:
        disparities = {}
        for emotion in emotions:
            pattern = f'{sensitive_attr}_([^_]+)_{metric}_{emotion}$'
            matching_cols = [col for col in row.index if re.match(pattern, col)]
            if len(matching_cols) >= 2:
                values = [row[col] for col in matching_cols]
                all_disparities = [abs(values[i] - values[j]) 
                                  for i in range(len(values)) 
                                  for j in range(i+1, len(values))]
                mean_disparity = sum(all_disparities) / len(all_disparities)
                disparities[emotion] = mean_disparity

    return disparities

def add_fairness_metrics_to_df(
        df, 
        emotions,
        sensitive_attr='gender', 
        fairness_name='statistical_parity', 
        run:int=None,
    ):
    if run is not None:
        df = df[df['run'] == run]
    fairness_definitions = {
        'statistical_parity': 'classwise_accuracy',
        'equal_opportunity': 'classwise_true_positive_rate',
        'equal_non_opportunity': 'classwise_false_positive_rate',
        'predictive_parity': 'classwise_positive_predictive_value',
        'negative_predictive_parity': 'classwise_negative_predictive_value',
        'overall_accuracy_equality': 'global_accuracy_weighted',
    }
    assert fairness_name in fairness_definitions, "Invalid fairness metric"

    metric = fairness_definitions[fairness_name]
    disparities = df.apply(lambda row: compute_fair_diff_row(row, emotions, sensitive_attr, metric), axis=1)

    if fairness_name == 'overall_accuracy_equality':
        df[fairness_name] = disparities
    else:
        for emotion in emotions:
            df[f'{fairness_name}_{emotion}'] = disparities.apply(lambda x: x.get(emotion, np.nan))
        df[fairness_name] = df[[f'{fairness_name}_{emotion}' for emotion in emotions]].mean(axis=1)

    return df