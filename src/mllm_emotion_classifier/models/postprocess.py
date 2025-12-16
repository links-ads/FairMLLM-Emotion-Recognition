import re
import logging
import Levenshtein

from typing import List, Optional

logger = logging.getLogger(__name__)

def postprocess_ser_response(
    class_labels: List[str],
    model_responses: List[str],
    threshold: float = 0.57
) -> List[Optional[str]]:
    results = []
    
    for model_response in model_responses:
        response_lower = model_response.strip().lower()
        matched = False
        
        for label in class_labels:
            if label.lower() in response_lower:
                results.append(label)
                matched = True
                break
        
        if matched:
            continue
        
        normalized = re.sub(r'[^\w\s]', ' ', model_response.lower())
        words = normalized.split()
        
        if not words:
            logger.warning(f'No valid words found in model response: "{model_response}"')
            results.append('Unknown')
            continue
        
        label_scores = {}
        for i, label in enumerate(class_labels):
            label_norm = re.sub(r'[^\w\s]', ' ', label.lower()).strip()
            scores = [Levenshtein.ratio(label_norm, word) for word in words]
            label_scores[i] = sum(s for s in scores if s >= threshold)
        
        if not any(label_scores.values()):
            logger.warning(f'Could not confidently parse response: "{model_response}"')
            results.append('Unknown')
        else:
            results.append(class_labels[max(label_scores, key=label_scores.get)])
    
    return results


# def _parse_emotion_response(self, responses: List[str]) -> List[str]:
#     parsed_emotions = []
    
#     for response in responses:
#         response = response.strip()
#         found_label = 'Unknown'
        
#         for label in self.class_labels:
#             if label.lower() in response.lower():
#                 found_label = label
#                 break
        
#         if found_label is None:
#             for letter, label in self.letter_to_label.items():
#                 if letter == response.upper():
#                     found_label = label
#                     break
        
#         if found_label == 'Unknown': 
#             logger.warning(f'Could not parse response: "{response}"')
        
#         parsed_emotions.append(found_label)
    
#     return parsed_emotions
