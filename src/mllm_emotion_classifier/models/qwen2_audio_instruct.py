"""Qwen2-Audio-Instruct wrapper for Emotion Recognition."""

import logging
import torch

from typing import List, Union
from .base import BaseEmotionModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, set_seed
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioCausalLMOutputWithPast
from .prompts.chat import build_conversation

logger = logging.getLogger(__name__)

DEFAULT_EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Fear", "Disgust", "Surprise"]

class Qwen2AudioInstructEmotionWrapper(BaseEmotionModel):
    
    def __init__(
        self,
        trust_remote_code: bool = True,
        torch_dtype: str = 'auto',
        max_new_tokens: int = 5,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        class_labels = None,
        prompt_name: str = "simple",
        device: str = "cuda",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        set_seed(seed)
        self.seed = seed

        self.name = "qwen2-audio-instruct"
        self.checkpoint = "Qwen/Qwen2-Audio-7B-Instruct"
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
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)

        self.class_labels = list(class_labels) if class_labels is not None else DEFAULT_EMOTIONS
        self.letter_to_label = {label[0].upper(): label for label in self.class_labels}

        self.prompt_name = prompt_name

    def collate_fn(self, inputs):
        input_audios = [_['audio'] for _ in inputs]
        labels = [_['label'] for _ in inputs]

        if "letter" in self.prompt_name:
            labels_str = ", ".join([f"{l}: {label}" for l, label in self.letter_to_label.items()])
        else:
            labels_str = ", ".join(self.class_labels)

        conversations = []
        for _ in input_audios:
            conversation = build_conversation(self.prompt_name, labels_str)
            conversations.append(conversation)

        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]

        inputs = self.processor(
            text=texts,
            audios=input_audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate
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

    def _parse_emotion_response(self, responses: List[str]) -> List[str]:
        parsed_emotions = []
        
        for response in responses:
            response = response.strip()
            found_label = None
            
            for label in self.class_labels:
                if label.lower() in response.lower():
                    found_label = label
                    break
            
            if found_label is None:
                for letter, label in self.letter_to_label.items():
                    if letter == response.upper():
                        found_label = label
                        break
            
            if found_label is None: 
                logger.warning(f'Could not parse response: "{response}"')
            
            parsed_emotions.append(found_label)
        
        return parsed_emotions

    def predict(self, inputs: dict) -> List[Union[str, int]]:
        set_seed(self.seed)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                do_sample=self.do_sample,
            )
        
        outputs = self._decode_outputs(inputs['input_ids'], output_ids)
        predictions = self._parse_emotion_response(outputs)
        return outputs

    def forward(self, inputs: dict) -> Qwen2AudioCausalLMOutputWithPast:
        return self.model(**inputs)