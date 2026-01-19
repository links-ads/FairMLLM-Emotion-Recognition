from .base import FairnessAnnotator


LANGUAGE = "English"

class MEADFairnessAnnotator(FairnessAnnotator):
    """Fairness preprocessor for MEAD dataset."""
    
    def __init__(self):
        super().__init__(dataset_name="mead", num_folds=1)
        self.LANGUAGE = LANGUAGE
    
    def _extract_gender_from_key(self, key: str) -> str:
        """
        Extract gender from MEAD key (e.g., 'mead-M027-neutral-level-1-026').
        If M is in the key name, the speaker is male. Otherwise, female.

        Parameters:
            key (str): The key string from which to extract gender.

        Returns:
            str: The gender of the speaker ('Male' or 'Female').
        """
        speaker = key.split('-')[1]
        if 'M' in speaker:
            return "Male"
        else:
            return "Female"
    
    def extract_sensitive_attributes(self, entry: dict) -> dict:
        """Extract sensitive attributes for MEAD."""
        key = entry.get('key', '')
        return {
            'gender': self._extract_gender_from_key(key),
            'language': self.LANGUAGE,
        }

if __name__ == "__main__":
    annotator = MEADFairnessAnnotator()
    annotator.add_sensitive_attributes_to_all_folds()