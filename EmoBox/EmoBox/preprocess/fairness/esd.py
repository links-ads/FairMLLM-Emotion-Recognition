from .base import FairnessAnnotator

class ESDFairnessAnnotator(FairnessAnnotator):
    """Fairness preprocessor for ESD dataset."""

    SPEAKER_INFO = {
        # Chinese speakers
        '0001': {'gender': 'Female', 'language': 'Chinese'},
        '0002': {'gender': 'Female', 'language': 'Chinese'},
        '0003': {'gender': 'Female', 'language': 'Chinese'},
        '0004': {'gender': 'Male', 'language': 'Chinese'},
        '0005': {'gender': 'Male', 'language': 'Chinese'},
        '0006': {'gender': 'Male', 'language': 'Chinese'},
        '0007': {'gender': 'Female', 'language': 'Chinese'},
        '0008': {'gender': 'Male', 'language': 'Chinese'},
        '0009': {'gender': 'Female', 'language': 'Chinese'},
        '0010': {'gender': 'Male', 'language': 'Chinese'},
        
        # English speakers
        '0011': {'gender': 'Male', 'language': 'English'},
        '0012': {'gender': 'Male', 'language': 'English'},
        '0013': {'gender': 'Male', 'language': 'English'},
        '0014': {'gender': 'Male', 'language': 'English'},
        '0015': {'gender': 'Female', 'language': 'English'},
        '0016': {'gender': 'Female', 'language': 'English'},
        '0017': {'gender': 'Female', 'language': 'English'},
        '0018': {'gender': 'Female', 'language': 'English'},
        '0019': {'gender': 'Male', 'language': 'English'},
        '0020': {'gender': 'Male', 'language': 'English'},
    }
    
    def __init__(self):
        super().__init__(dataset_name="esd", num_folds=5)
    
    def _extract_speaker_id(self, key: str) -> str:
        """Extract speaker ID from ESD key (e.g., 'esd-0005-000001' -> '0005')."""
        parts = key.split('-')
        if len(parts) >= 2:
            return parts[1]
        return None
    
    def extract_sensitive_attributes(self, entry: dict) -> dict:
        """Extract sensitive attributes for ESD."""
        key = entry.get('key', '')
        speaker_id = self._extract_speaker_id(key)
        info = self.SPEAKER_INFO[speaker_id]
        return {
            'gender': info['gender'],
            'language': info['language'],
        }

if __name__ == "__main__":
    annotator = ESDFairnessAnnotator()
    annotator.add_sensitive_attributes_to_all_folds()