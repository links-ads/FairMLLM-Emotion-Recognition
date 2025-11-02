"""Qwen2-Audio wrapper for Emotion Recognition."""

import logging
import torch

from typing import List, Union
from .base import BaseEmotionModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.models.qwen2_audio.modeling_qwen2_audio  import Qwen2AudioCausalLMOutputWithPast


logger = logging.getLogger(__name__)


class Qwen2AudioEmotionWrapper(BaseEmotionModel):
    
    # DEFAULT_PROMPT = "<|audio_bos|><|AUDIO|><|audio_eos|>What emotion is expressed in this audio? Answer with a single word emotion label."

    AUDIO_PROMPT_TEMPLATE = (
        "<|audio_bos|><|AUDIO|><|audio_eos|>"
        "What emotion is expressed in this audio? "
        "Answer with a single word emotion label among: {labels}."
    )

    DEFAULT_EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Fear", "Disgust", "Surprise"]
    
    def __init__(
        self, 
        # config: ModelConfig,
        checkpoint: str = "Qwen/Qwen2-Audio-7B",
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        class_labels = None,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.do_sample = do_sample
        self.device = device

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.checkpoint,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype
        )
        self.model.eval()
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        self.processor.tokenizer.padding_side = 'left'

        self.class_labels = class_labels if class_labels is not None else self.DEFAULT_EMOTIONS

    def collate_fn(self, inputs):
        input_audios = [_['audio'] for _ in inputs]
        input_texts = [self.AUDIO_PROMPT_TEMPLATE.format(labels=", ".join(self.class_labels)) for _ in inputs]
        labels = [_['label'] for _ in inputs]
        inputs = self.processor(
            text=input_texts,
            audio=input_audios,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        return inputs, labels
    
    def _decode_outputs(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> List[str]:
        output_ids = output_ids[:, input_ids.size(1):]
        outputs = self.processor.batch_decode(
            output_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        return outputs
    
    def _parse_emotion_response(
        self,
        responses: List[str],
    ) -> str:
        # emotions = []
        # for response in responses:
        #     emotion = parse_emotion_response(response)
        #     emotions.append(emotion)
        # return emotions
        return responses

    def predict(
        self,
        inputs: dict,
    ) -> List[Union[str, int]]:
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            do_sample=self.do_sample,
        )
        outputs = self._decode_outputs(
            inputs['input_ids'],
            output_ids
        )
        predictions = self._parse_emotion_response(outputs)
        return predictions

    def forward(
        self,
        inputs: dict,
    ) -> Qwen2AudioCausalLMOutputWithPast:
        output = self.model(**inputs)
        return output
