import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

class ClassificationMetrics(object):
    def __init__( self, y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[str]=None):
        assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match."
        assert len(y_true.shape) == 1, "y_true and y_pred must be 1-dimensional arrays."
        assert len(np.unique(y_true)) > 1, "y_true must contain more than one class."

        self.y_true = y_true
        self.y_pred = y_pred
        
        self.classes = np.unique(y_true).tolist()
        self.class_labels = class_labels if class_labels is not None else self.classes

        self.confusion_matrices = {cls: self._set_confusion_matrix(cls) for cls in self.classes}
        self.global_confusion_matrix = confusion_matrix(
            y_true, y_pred, 
            labels=class_labels if class_labels is not None else self.classes
        )
    
    def _set_confusion_matrix(self, cls):
        y_true_binary = (self.y_true == cls).astype(int)
        y_pred_binary = (self.y_pred == cls).astype(int)
        return confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    
    def _parse_metric(self, results: dict, per_class: bool):
        if per_class:
            return {cls: round(float(results[cls]), 4) for cls in self.classes}
        else:
            return round(float(np.mean(list(results.values()))), 4)
    
    def _compute_metric(self, metric_fn, per_class: bool):
        results = {}
        for cls in self.classes:
            tn, fp, fn, tp = self.confusion_matrices[cls]
            results[cls] = metric_fn(tn, fp, fn, tp)
        return self._parse_metric(results, per_class)

    def acceptance_rate(self, per_class=True):
        """Acceptance Rate = (TP + FP) / Total"""
        total = len(self.y_true)
        return self._compute_metric(
            lambda tn, fp, fn, tp: (tp + fp) / total if total > 0 else 0.0,
            per_class
        )
    
    def accuracy(self, per_class=True):
        """Accuracy = (TP + TN) / (TP + TN + FP + FN)"""
        return self._compute_metric(
            lambda tn, fp, fn, tp: (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0,
            per_class
        )
    
    def false_positive_rate(self, per_class=True):
        """FPR = FP / (FP + TN)"""
        return self._compute_metric(
            lambda tn, fp, fn, tp: fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            per_class
        )
    
    def false_negative_rate(self, per_class=True):
        """FNR = FN / (FN + TP)"""
        return self._compute_metric(
            lambda tn, fp, fn, tp: fn / (fn + tp) if (fn + tp) > 0 else 0.0,
            per_class
        )
    
    def true_positive_rate(self, per_class=True):
        """TPR (Recall) = TP / (TP + FN)"""
        return self._compute_metric(
            lambda tn, fp, fn, tp: tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            per_class
        )
    
    def true_negative_rate(self, per_class=True):
        """TNR = TN / (TN + FP)"""
        return self._compute_metric(
            lambda tn, fp, fn, tp: tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            per_class
        )
    
    def positive_predictive_value(self, per_class=True):
        """PPV (Precision) = TP / (TP + FP)"""
        return self._compute_metric(
            lambda tn, fp, fn, tp: tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            per_class
        )
    
    def negative_predictive_value(self, per_class=True):
        """NPV = TN / (TN + FN)"""
        return self._compute_metric(
            lambda tn, fp, fn, tp: tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            per_class
        )

    def f1(self, per_class=True):
        """F1 Score = 2TP / (2TP + FP + FN)"""
        return self._compute_metric(
            lambda tn, fp, fn, tp: (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
            per_class
        )

    def accuracy_weighted(self):
        """Accuracy (overall correctness)"""
        return round(float(accuracy_score(self.y_true, self.y_pred)), 4)

    def accuracy_unweighted(self):
        """Balanced Accuracy (average of per-class recalls)"""
        return round(float(balanced_accuracy_score(self.y_true, self.y_pred)), 4)
    
    def precision_macro(self):
        """Macro Precision"""
        return round(float(precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)), 4)
    
    def precision_weighted(self):
        """Weighted Precision"""
        return round(float(precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)), 4)
    
    def recall_macro(self):
        """Macro Recall"""
        return round(float(recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)), 4)
    
    def recall_weighted(self):
        """Weighted Recall"""
        return round(float(recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)), 4)
    
    def f1_macro(self):
        """Macro F1 Score"""
        return round(float(f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)), 4)
    
    def f1_weighted(self):
        """Weighted F1 Score"""
        return round(float(f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)), 4)
    
    def compute(self, sens_attr_dict: dict[str, np.ndarray]=None):
        metrics = {
            'global':{
                'f1_macro': self.f1_macro(),
                'f1_weighted': self.f1_weighted(),
                'accuracy_unweighted': self.accuracy_unweighted(),
                'accuracy_weighted': self.accuracy_weighted(),
                'precision_macro': self.precision_macro(),
                'precision_weighted': self.precision_weighted(),
                'recall_macro': self.recall_macro(),
                'recall_weighted': self.recall_weighted(),
            },
            'classwise':{
                'accuracy': self.accuracy(per_class=True),
                'false_positive_rate': self.false_positive_rate(per_class=True),
                'false_negative_rate': self.false_negative_rate(per_class=True),
                'true_positive_rate': self.true_positive_rate(per_class=True),
                'true_negative_rate': self.true_negative_rate(per_class=True),
                'positive_predictive_value': self.positive_predictive_value(per_class=True),
                'negative_predictive_value': self.negative_predictive_value(per_class=True),
                'f1_score': self.f1(per_class=True),
            }
        }
        if sens_attr_dict is not None:
            for attr_name, sens_attr in sens_attr_dict.items():
                metrics[attr_name] = {}
                for cls in np.unique(sens_attr):
                    idx = sens_attr == cls
                    y_true_subset = self.y_true[idx]
                    y_pred_subset = self.y_pred[idx]
                    sub_calculator = ClassificationMetrics(y_true_subset, y_pred_subset)
                    metrics[attr_name][str(cls)] = sub_calculator.compute()
        return metrics