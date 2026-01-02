from .base import FairnessAnnotator


LANGUAGE = "English"

class IEMOCAPFairnessAnnotator(FairnessAnnotator):
    """Fairness preprocessor for IEMOCAP dataset."""
    
    def __init__(self):
        super().__init__(dataset_name="iemocap", num_folds=5)
    
    def _extract_gender_from_key(self, key: str) -> str:
        """Extract gender from IEMOCAP key (e.g., 'Ses01F_impro01_F000')."""
        gender_code = key.split('_')[-1][0]
        return 'Female' if gender_code == 'F' else 'Male'
    
    def extract_sensitive_attributes(self, entry: dict) -> dict:
        """Extract sensitive attributes for IEMOCAP."""
        key = entry.get('key', '')
        return {
            'gender': self._extract_gender_from_key(key),
            'language': LANGUAGE,
        }

if __name__ == "__main__":
    annotator = IEMOCAPFairnessAnnotator()
    annotator.add_sensitive_attributes_to_all_folds()