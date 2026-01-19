"""AudioFlamingo3 wrapper for Emotion Recognition."""

import logging
import torch

from typing import List, Union
from .base import BaseEmotionModel
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor, set_seed
from .prompts.chat_flamingo import build_conversation
from .postprocess import postprocess_ser_response

logger = logging.getLogger(__name__)

DEFAULT_EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Fear", "Disgust", "Surprise"]


class AudioFlamingo3EmotionWrapper(BaseEmotionModel):
    
    def __init__(
        self,
        trust_remote_code: bool = True,
        torch_dtype: str = 'auto',
        max_new_tokens: int = 5,
        min_new_tokens: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        class_labels = None,
        prompt_name: str = "user_labels",
        device: str = "cuda",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        set_seed(seed)
        self.seed = seed

        self.name = "audio-flamingo-3"
        self.checkpoint = "nvidia/audio-flamingo-3-hf"
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.do_sample = do_sample
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.checkpoint,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype
        )
        self.model.eval()
        # self.model.to(self.device)

        self.class_labels = list(class_labels) if class_labels is not None else DEFAULT_EMOTIONS
        self.letter_to_label = {label[0].upper(): label for label in self.class_labels}

        self.prompt_name = prompt_name

    def collate_fn(self, inputs):
        input_audios = [_['audio'] for _ in inputs]
        labels = [_['label'] for _ in inputs]

        labels_str = ", ".join(self.class_labels)

        conversations = []
        for _ in input_audios:
            conversation = build_conversation(self.prompt_name, labels_str)
            conversations.append(conversation)

        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]

        processed_inputs = self.processor(
            text=texts,
            audio=input_audios,
            return_tensors="pt",
            tokenize=True,
            # add_generation_prompt=True,
            # return_dict=True,
        )  # Remove .to(self.model.device) - tensors will be moved in evaluation loop
        
        return processed_inputs, labels
    
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

    def predict(self, inputs: dict) -> List[Union[str, None]]:
        set_seed(self.seed)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        outputs = self._decode_outputs(inputs['input_ids'], output_ids)
        predictions = postprocess_ser_response(
            class_labels=self.class_labels,
            model_responses=outputs,
        )
        return predictions

    def forward(self, inputs: dict):
        return self.model(**inputs)