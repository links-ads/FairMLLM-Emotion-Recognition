from .qwen2_audio import Qwen2AudioEmotionWrapper
from .base import BaseEmotionModel

class ModelFactory:
    def create(name: str, **kwargs) -> BaseEmotionModel:
        if name == "qwen2-audio":
            model = Qwen2AudioEmotionWrapper(**kwargs)
        else:
            raise ValueError(f"Model '{name}' not found.")
        return model