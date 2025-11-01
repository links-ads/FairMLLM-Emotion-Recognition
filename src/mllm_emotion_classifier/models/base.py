import torch

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
    
class BaseEmotionModel(ABC):
    """Abstract base class for emotion recognition models."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.processor = None
        
    @abstractmethod
    def collate_fn(
        self,
        inputs: List[Dict[str, Any]],
        processor: Any
    ) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def predict(
        self, 
        inputs: dict[str, torch.Tensor]
    ) -> Union[Union[str, int]]:
        """The inputs should be the output of the collate_fn method."""
        pass