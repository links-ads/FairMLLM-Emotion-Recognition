import re
import numpy as np

def compute_fair_diff_row(row, sensitive_attr='gender', metric='classwise_accuracy'):
    assert metric in ['classwise_accuracy', # Statistical Parity
                      'classwise_true_positive_rate', # Equal Opportunity
                      'classwise_false_positive_rate',  # Equal Non-Opportunity
                      'classwise_positive_predictive_value', # Predictive Parity
                      'classwise_negative_predictive_value'], "Invalid metric" # Negative Predictive Parity
    emotions = ['Angry', 'Neutral', 'Sad', 'Happy']
    
    disparities = {}
    
    for emotion in emotions:
        pattern = f'{sensitive_attr}_([^_]+)_{metric}_{emotion}$'
        matching_cols = [col for col in row.index if re.match(pattern, col)]

        if len(matching_cols) >= 2:
            values = [row[col] for col in matching_cols]
            disparity = abs(values[0] - values[1])
            disparities[emotion] = disparity

    return disparities

def add_fairness_metrics_by_emotion(
        df, emotions:list=['Angry', 'Neutral', 'Sad', 'Happy'],
        sensitive_attr='gender', fairness_name='statistical_parity', run:int=None,
    ):
    if run is not None:
        df = df[df['run'] == run]
    fairness_definitions = {
        'statistical_parity': 'classwise_accuracy',
        'equal_opportunity': 'classwise_true_positive_rate',
        'equal_non_opportunity': 'classwise_false_positive_rate',
        'predictive_parity': 'classwise_positive_predictive_value',
        'negative_predictive_parity': 'classwise_negative_predictive_value'
    }
    assert fairness_name in fairness_definitions, "Invalid fairness metric"

    metric = fairness_definitions[fairness_name]
    disparities = df.apply(lambda row: compute_fair_diff_row(row, sensitive_attr, metric), axis=1)

    for emotion in emotions:
        df[f'{fairness_name}_{emotion}'] = disparities.apply(lambda x: x.get(emotion, np.nan))
    
    df[fairness_name] = df[[f'{fairness_name}_{emotion}' for emotion in emotions]].mean(axis=1)

    return df