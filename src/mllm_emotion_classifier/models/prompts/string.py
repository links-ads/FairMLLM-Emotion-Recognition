PROMPTS = {
    "simple": "<|audio_bos|><|AUDIO|><|audio_eos|>Classify the speaker's tone. Options: {labels}. Answer:",
    # "letter": "<|audio_bos|><|AUDIO|><|audio_eos|>Emotion? {labels}. Letter only:",
    "task": "<|audio_bos|><|AUDIO|><|audio_eos|>Task: Speech Emotion Recognition. Options: {labels}. Answer:",
}

def get_prompt_template(prompt_name: str) -> str:
    return PROMPTS.get(prompt_name, PROMPTS["simple"])