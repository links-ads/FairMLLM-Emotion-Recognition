# MLLM Emotion Classifier

A clean, modular framework for emotion recognition using Multimodal Large Language Models (MLLMs).

## Features

- 🏭 **Model Factory Pattern**: Easily switch between different MLLM models
- 🎯 **Simple API**: Intuitive interface for emotion prediction
- 📦 **Batch Processing**: Efficient processing of multiple audio files
- 🔌 **Extensible**: Easy to add new models and custom configurations
- 🌐 **URL Support**: Load audio from local files or remote URLs

## Quick Start

### Basic Usage

```python
from mllm_emotion_classifier import create_emotion_model

# Create model
model = create_emotion_model("qwen2-audio")

# Predict emotion
prediction = model.predict("path/to/audio.wav")
print(f"Emotion: {prediction.emotion}")
```

### Batch Processing

```python
# Process multiple files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
predictions = model.predict_batch(audio_files)

for audio, pred in zip(audio_files, predictions):
    print(f"{audio}: {pred.emotion}")
```

### Custom Configuration

```python
from mllm_emotion_classifier import ModelFactory, ModelConfig

config = ModelConfig(
    model_name="qwen2-audio",
    device="cuda",
    batch_size=4,
    max_new_tokens=128
)

model = ModelFactory.create_model("qwen2-audio", config=config)
```

### Custom Prompt

```python
custom_prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>What emotion is this? (happy/sad/angry/neutral):"
prediction = model.predict("audio.wav", prompt=custom_prompt)
```

## Architecture

```
mllm_emotion_classifier/
├── __init__.py          # Package exports
├── base.py              # Base classes and interfaces
├── factory.py           # Model factory
├── models/              # Model implementations
│   ├── __init__.py
│   └── qwen2_audio.py   # Qwen2-Audio implementation
├── utils.py             # Utility functions
└── examples.py          # Usage examples
```

## Available Models

- **qwen2-audio**: Qwen2-Audio-7B (default)
- **qwen2-audio-7b**: Qwen2-Audio-7B
- **qwen2-audio-instruct**: Qwen2-Audio-7B-Instruct

List all available models:

```python
from mllm_emotion_classifier import ModelFactory

models = ModelFactory.list_models()
print(models)
```

## Adding Custom Models

### 1. Implement the Base Class

```python
from mllm_emotion_classifier.base import BaseEmotionModel, EmotionPrediction

class MyCustomModel(BaseEmotionModel):
    def load(self):
        # Load your model
        pass
    
    def predict(self, audio_path, prompt=None):
        # Implement prediction
        pass
    
    def predict_batch(self, audio_paths, prompts=None):
        # Implement batch prediction
        pass
```

### 2. Register the Model

```python
from mllm_emotion_classifier import ModelFactory

ModelFactory.register_model("my-model", MyCustomModel)
ModelFactory.register_checkpoint("my-model", "path/to/checkpoint")
```

### 3. Use It

```python
model = create_emotion_model("my-model")
```

## Integration with EmoBox

```python
from mllm_emotion_classifier import create_emotion_model
from EmoBox.EmoBox.EmoDataset import EmoDataset

# Load dataset
dataset = EmoDataset(
    dataset="iemocap",
    data_dir="EmoBox/data/",
    meta_data_dir="EmoBox/data/",
    fold=1,
    split="test"
)

# Create model
model = create_emotion_model("qwen2-audio")

# Evaluate
for sample in dataset:
    prediction = model.predict(sample['audio_path'])
    print(f"Predicted: {prediction.emotion}, Ground Truth: {sample['label']}")
```

## EmotionPrediction Object

The `predict()` method returns an `EmotionPrediction` object with:

- `emotion`: Extracted emotion label (string)
- `raw_output`: Raw model output (string)
- `confidence`: Confidence score (optional, float)
- `logits`: Model logits (optional, torch.Tensor)
- `metadata`: Additional metadata (dict)

```python
prediction = model.predict("audio.wav")

print(prediction.emotion)      # 'happy'
print(prediction.raw_output)   # 'The emotion is: happy'
print(prediction.metadata)     # {'audio_path': 'audio.wav'}
```

## Requirements

See `requirements.txt` in the EmoBox folder for dependencies:
- torch
- transformers
- torchaudio
- librosa
- requests
- scikit-learn

## License

See the LICENSE file in the project root.
