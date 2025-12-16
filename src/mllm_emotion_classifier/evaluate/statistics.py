import numpy as np

class Statistics(object):
    def __init__(self, y_true: np.ndarray, sens_attr_dict: dict[str, np.ndarray]=None):
        self.y_true = y_true
        self.sens_attr_dict = sens_attr_dict
        self.classes = np.unique(y_true)
    
    def _count_samples(self, mask=None):
        """Count samples with optional mask"""
        return int(np.sum(mask) if mask is not None else len(self.y_true))
    
    def _group_by(self, values, mask=None):
        """Count samples grouped by unique values"""
        base_mask = mask if mask is not None else np.ones(len(values), dtype=bool)
        return {str(val): self._count_samples(base_mask & (values == val)) 
                for val in np.unique(values)}
    
    def _cross_tabulate(self, attr_values):
        """Cross-tabulate class counts by sensitive attribute values"""
        return {str(val): self._group_by(self.y_true, attr_values == val) 
                for val in np.unique(attr_values)}
    
    def compute(self):
        """Compute all statistics"""
        stats = {
            'total_samples': self._count_samples(),
            'samples_by_class': self._group_by(self.y_true),
        }
        
        if self.sens_attr_dict:
            stats['samples_by_sensitive_attr'] = {
                str(name): self._group_by(vals) 
                for name, vals in self.sens_attr_dict.items()
            }
            stats['samples_by_class_and_sensitive_attr'] = {
                str(name): self._cross_tabulate(vals) 
                for name, vals in self.sens_attr_dict.items()
            }
        
        return stats