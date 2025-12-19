import os
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from .metrics import ClassificationMetrics
from .statistics import Statistics
from ..utils import quantile_binning


class Evaluator:

    def __init__(self, output_dir=None):

        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        self.y_true = None
        self.y_pred = None
        self.sens_attr_dict = None

        self.metrics_obj = None
        self.fairness_obj = None

        self.results = None
    
    def _collect_predictions(self, model, dataloader, n_samples=None):
        y_pred, y_true = [], []
        sens_attr_dict = defaultdict(list)
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Inference")):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            preds = model.predict(inputs)
            y_pred.extend(preds)
            y_true.extend(labels)
            start_idx = batch_idx * dataloader.batch_size
            for idx in range(start_idx, min(start_idx + len(labels), len(dataloader.dataset))):
                sample = dataloader.dataset[idx]
                for k, v in sample.items():
                    if k not in {'audio', 'label', 'key', 'text'}:
                        sens_attr_dict[k].append(v)
            if n_samples is not None and len(y_pred) >= n_samples:
                break
                        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sens_attr_dict = {k: np.array(v) for k, v in sens_attr_dict.items()}

        # binning sensitive attributes if needed
        for attr_name, attr_values in sens_attr_dict.items():
            if attr_name.lower() in ['age']:
                binned_values = quantile_binning(attr_values, n_bins=4)
                sens_attr_dict[attr_name] = binned_values
                print(f"Binned sensitive attribute '{attr_name}' into quantile bins.")
                
        return y_true, y_pred, sens_attr_dict     
    
    def _compute_metrics(self, y_true, y_pred, sens_attr_dict):
        metrics = {}

        self.metrics_obj = ClassificationMetrics(y_true, y_pred)
        metrics['metrics'] = self.metrics_obj.compute(sens_attr_dict=sens_attr_dict)
        
        # if sens_attr_dict is None:
        #     return metrics
        
        # metrics['fairness'] = {}
        # for attr_name, sens_attr_values in sens_attr_dict.items():
        #     print(type(sens_attr_values))
        #     print(type(sens_attr_values[0]))
        #     self.fairness_obj = FairnessMetrics(
        #         y_true, y_pred, attr_name, sens_attr_values
        #     )
        #     metrics['fairness'][attr_name] = self.fairness_obj.compute()
            
        return metrics

    def _compute_stats(self, y_true, sens_attr_dict=None):
        stats = Statistics(y_true, sens_attr_dict).compute()
        return {'stats': stats}
    
    def evaluate(self, model, dataloader, n_samples=None, fold=None):
        model_name = model.name
        dataset_name = dataloader.dataset.name
        class_labels = set(dataloader.dataset.label_map.values())

        print("\n" + "="*80)
        print(f"Evaluating {model_name} on {dataset_name}")
        print("="*80)
        
        self.y_true, self.y_pred, self.sens_attr_dict = self._collect_predictions(model, dataloader, n_samples=n_samples)

        total_indices = list(range(len(self.y_pred)))
        valid_indices = [i for i, p in enumerate(self.y_pred) if p is not "Unknown"]
        if not valid_indices:
            raise ValueError("No valid predictions were made.")
        
        # self.y_pred = np.array([self.y_pred[i] for i in valid_indices])
        # self.y_true = np.array([self.y_true[i] for i in valid_indices])
        # if self.sens_attr_dict:
        #     self.sens_attr_dict = {
        #         k: np.array([v[i] for i in valid_indices]) 
        #         for k, v in self.sens_attr_dict.items()
        #     }

        # stats = self._compute_stats(self.y_pred, self.sens_attr_dict)
        metrics = self._compute_metrics(self.y_true, self.y_pred, self.sens_attr_dict)
        
        self.results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': dataset_name,
            'model_name': model_name,
            'fold': fold,
            'num_samples': len(self.y_pred),
            'valid_rate': round(len(valid_indices) / len(total_indices), 4),
            'class_labels': list(class_labels),
            # **stats,
            **metrics,
        }
        return self.results
    
    def print_results(self):
        assert self.results is not None, "No results to print. Please run evaluate() first."
        
        print("\nEvaluation Results:")
        print("-" * 80)
        print(f"Dataset: {self.results['dataset']}")
        print(f"Model: {self.results['model_name']}")
        print(f"Fold: {self.results['fold']}")
        print(f"Number of Samples: {self.results['num_samples']}")
        print("\nOverall Metrics:")
        for metric, value in self.results['metrics']['overall'].items():
            print(f"  {metric}: {value:.4f}")
        print("\nPer-Class Metrics:")
        for metric, values in self.results['metrics'].items():
            if metric != 'overall':
                print(f"  {metric}:")
                for cls, value in values.items():
                    print(f"    {cls}: {value:.4f}")
        print("\nFairness Metrics:")
        for attr, metrics in self.results['fairness'].items():
            print(f"  Sensitive Attribute: {attr}")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
        print("-" * 80)
