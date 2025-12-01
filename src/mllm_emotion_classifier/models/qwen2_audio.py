"""Qwen2-Audio wrapper for Emotion Recognition."""

import logging
import torch
import random
import numpy as np

from typing import List, Union
from .base import BaseEmotionModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, set_seed
from transformers.models.qwen2_audio.modeling_qwen2_audio  import Qwen2AudioCausalLMOutputWithPast


logger = logging.getLogger(__name__)


# def set_seed(seed: int = 42):
#     """Set seeds for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

class Qwen2AudioEmotionWrapper(BaseEmotionModel):

    AUDIO_PROMPT_TEMPLATE = (
        "<|audio_bos|><|AUDIO|><|audio_eos|>"
        "Classify the speaker’s tone in the audio. "
        "Select one of: {labels}. "
        "Answer:"
    )


    DEFAULT_EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Fear", "Disgust", "Surprise"]
    
    def __init__(
        self, 
        # config: ModelConfig,
        checkpoint: str = "Qwen/Qwen2-Audio-7B",
        trust_remote_code: bool = True,
        torch_dtype: str = 'float16',
        max_new_tokens: int = 5,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        class_labels = None,
        device: str = "auto",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        set_seed(seed)
        self.seed = seed

        self.name = checkpoint.split("/")[-1]
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

        self.class_labels = list(class_labels) if class_labels is not None else self.DEFAULT_EMOTIONS

    def collate_fn(self, inputs):
        input_audios = [_['audio'] for _ in inputs]

        labels = ", ".join(self.class_labels)
        # label_dict = {label[0]: label for label in self.class_labels}
        input_texts = [self.AUDIO_PROMPT_TEMPLATE.format(labels=labels) for _ in inputs]
        
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

    def _parse_emotion_response(self, responses: List[str]) -> List[str]:
        parsed_emotions = []
        
        for response in responses:
            response = response.strip()
            found_label = None
            for label in self.class_labels:
                if label.lower() in response.lower():
                    found_label = label.capitalize()
            if found_label is None: 
                print(f'Warning: could not parse response "{response}"')
            parsed_emotions.append(found_label)

        # parsed_emotions = []
        # for response in responses:
        #     response = response.strip().upper()
        #     found_label = self.letter_to_label.get(response)
        #     parsed_emotions.append(found_label)
        
        return parsed_emotions

    def predict(
        self,
        inputs: dict,
    ) -> List[Union[str, int]]:
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                do_sample=self.do_sample,
                # num_beams=1,
                temperature=0.00000000001,  # Keep at 1.0 when do_sample=False
                top_p=0.00000000001,
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
