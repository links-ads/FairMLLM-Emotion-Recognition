CHAT_PROMPTS = {
    "system_labels": {
        "system": "You are a highly capable assistant specialized in Speech Emotion Recognition. You receive audio input and identify the emotion expressed in the speech. The emotion must be one of the following categories: {labels}.",
        "user": "Classify the tone of the speaker in the preceding audio. Answer:"
    },
    "user_labels": {
        "system": "You are a highly capable assistant specialized in Speech Emotion Recognition",
        "user": "Classify the speaker's tone in the preceding audio. Options: {labels}. Answer:"
    },
    # "user_labels": {
    #     "system": "You are a highly capable assistant specialized in Speech Emotion Recognition.",
    #     "user": "Classify the speaker's tone in the preceding audio in French. Options: {labels}. Answer:"
    # },
    # "user_labels": {
    #     "system": "Vous êtes un assistant hautement compétent, spécialisé dans la reconnaissance des émotions dans la parole",
    #     "user": "Classez le ton de l’orateur dans l’audio précédent. Options : {labels}. Réponse:"
    # },
    "direct": {
        "system": None,
        "user": "Listen to the audio and identify the speaker's emotion. Choose from: {labels}. Answer:"
    },
    "cameo": {
        "system": "You are an audio emotion recognizer.",
        "user": "Based only on the tone , pitch , rhythm , and intonation of the speaker in the audio , identify the speaker ’s emotional state. Respond with exactly one word from the following list : {labels}. Do not explain . Do not use synonyms ."
    }
}

def get_chat_prompt(prompt_name: str) -> dict:
    return CHAT_PROMPTS.get(prompt_name, CHAT_PROMPTS["user_labels"])

def build_conversation(prompt_name: str, labels_str: str) -> list:
    prompt = get_chat_prompt(prompt_name)
    
    user_content = user_content = prompt["user"].format(labels=labels_str) if "{labels}" in prompt["user"] else prompt["user"]
    
    conversation = []
    if prompt["system"] is not None:
        system_content = prompt["system"].format(labels=labels_str) if "{labels}" in prompt["system"] else prompt["system"]
        conversation.append({'role': 'system', 'content': system_content})

    conversation.append({
        'role': 'user',
        'content': [
            {'type': 'audio', 'audio_url': 'placeholder'},
            {'type': 'text', 'text': user_content}
        ]
    })
    
    return conversation