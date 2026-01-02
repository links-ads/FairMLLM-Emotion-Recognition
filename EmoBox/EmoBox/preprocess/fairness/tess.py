from .base import FairnessAnnotator


LANGUAGE = "English"
GENDER = "Female"

class TESSFairnessAnnotator(FairnessAnnotator):
    """Fairness preprocessor for TESS dataset."""
    
    def __init__(self):
        super().__init__(dataset_name="tess", num_folds=1)
    
    def _extract_age_from_key(self, key: str) -> str:
        """Extract age from TESS key (e.g., 'Ses01F_impro01_F000')."""
        age = '64' if "OAF" in key else '26'
        return age
    
    def extract_sensitive_attributes(self, entry: dict) -> dict:
        """Extract sensitive attributes for TESS."""
        key = entry.get('key', '')
        return {
            'agegroup': self._extract_age_from_key(key),
            'gender': GENDER,
            'language': LANGUAGE,
        }

if __name__ == "__main__":
    annotator = TESSFairnessAnnotator()
    annotator.add_sensitive_attributes_to_all_folds()