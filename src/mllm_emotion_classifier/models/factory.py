from .qwen2_audio import Qwen2AudioEmotionWrapper
from .qwen2_audio_instruct import Qwen2AudioInstructEmotionWrapper
from .base import BaseEmotionModel

class ModelFactory:
    def create(name: str, **kwargs) -> BaseEmotionModel:
        if name == "qwen2-audio":
            model = Qwen2AudioEmotionWrapper(checkpoint="Qwen/Qwen2-Audio-7B", **kwargs)
        elif name == "qwen2-audio-instruct":
            model = Qwen2AudioInstructEmotionWrapper(checkpoint="Qwen/Qwen2-Audio-7B-Instruct", **kwargs)
        else:
            raise ValueError(f"Model '{name}' not found.")
        return model