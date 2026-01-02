from .base import FairnessAnnotator


LANGUAGE = "English"

class RAVDESSFairnessAnnotator(FairnessAnnotator):
    """Fairness preprocessor for RAVDESS dataset."""
    
    def __init__(self):
        super().__init__(dataset_name="ravdess", num_folds=6)
    
    def _extract_gender_from_key(self, key: str) -> str:
        """Extract gender from RAVDESS key (e.g., '03-01-01-01-01-01-11.wav')."""
        gender_code = key.split('-')[-1]
        return 'Female' if int(gender_code) % 2 == 0 else 'Male'
    
    def extract_sensitive_attributes(self, entry: dict) -> dict:
        """Extract sensitive attributes for RAVDESS."""
        key = entry.get('key', '')
        return {
            'gender': self._extract_gender_from_key(key),
            'language': LANGUAGE,
        }

if __name__ == "__main__":
    annotator = RAVDESSFairnessAnnotator()
    annotator.add_sensitive_attributes_to_all_folds()