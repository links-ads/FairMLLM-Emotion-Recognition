CHAT_PROMPTS = {
    "simple": {
        "system": "You are a highly capable assistant specialized in Speech Emotion Recognition (SER). You receive audio input and identify the emotion expressed in the speech. The emotion must be one of the following categories: {labels}.",
        "user": "Classify the tone of the speaker in the preceding audio."
    },
    "format": {
        "system": "You are a highly capable assistant specialized in Speech Emotion Recognition (SER). You receive audio input and identify the emotion expressed in the speech. Return your answer in the following format: '| Emotion: <emotion> |'. The emotion must be one of the following categories: {labels}.",
        "user": "Classify the tone of the speaker in the preceding audio."
    },
    "user_labels": {
        "system": "You are a highly capable assistant specialized in Speech Emotion Recognition (SER)",
        "user": "Classify the tone of the speaker in the preceding audio. The possible emotions are: {labels}. Answer:"
    }
}


def get_chat_prompt(prompt_name: str) -> dict:
    return CHAT_PROMPTS.get(prompt_name, CHAT_PROMPTS["simple"])

def build_conversation(prompt_name: str, labels_str: str) -> list:
    prompt = get_chat_prompt(prompt_name)

    if prompt_name == "user_labels":
        prompt["user"] = prompt["user"].format(labels=labels_str)
    else:
        prompt["system"] = prompt["system"].format(labels=labels_str)

    return [
        {'role': 'system', 'content': prompt["system"]},
        {'role': 'user', 'content': [
            {'type': 'audio', 'audio_url': '|AUDIO|'},
            {'type': 'text', 'text': prompt["user"]}
        ]}
    ]