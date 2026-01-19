import pandas as pd

from .base import FairnessAnnotator

LANGUAGE = "English"
SPEAKER_INFO = {
    "Monica": {
        "gender": "Female",
        "age": 29
    },
    "Rachel": {
        "gender": "Female",
        "age": 24
    },
    "Phoebe": {
        "gender": "Female",
        "age": 30
    },
    "Ross": {
        "gender": "Male",
        "age": 26
    },
    "Joey": {
        "gender": "Male",
        "age": 26
    },
    "Chandler": {
        "gender": "Male",
        "age": 23
    }
}

UTTERANCES_TEST_INFO_FILE_PATH = "downloads/meld/test_sent_emo.csv"
UTTERANCES_TRAIN_INFO_FILE_PATH = "downloads/meld/train_sent_emo.csv"
UTTERANCES_DEV_INFO_FILE_PATH = "downloads/meld/dev_sent_emo.csv"

class MELDFairnessAnnotator(FairnessAnnotator):
    """Fairness preprocessor for MELD dataset."""
    
    def __init__(self):
        super().__init__(dataset_name="meld", num_folds=1)
        self.LANGUAGE = LANGUAGE
        self.utterance_info = self._read_utterance_info_files()

    def _read_utterance_info_files(self):
        """
        Reads the utterance info CSV files (train, dev, test) and returns a dictionary mapping utterance ID to the
        utterance info. The files are composed of the following columns: 
        
        Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime. 
        
        They are separated by ','. Since utterance id repeats accross different splits, we keep each split in a different dataframe.

        Returns:
            dict: A dictionary where keys are train, test, dev and values are dataframes with utterance info.
        """

        train_df = pd.read_csv(UTTERANCES_TRAIN_INFO_FILE_PATH)
        dev_df = pd.read_csv(UTTERANCES_DEV_INFO_FILE_PATH)
        test_df = pd.read_csv(UTTERANCES_TEST_INFO_FILE_PATH)
        return {
            'train': train_df,
            'dev': dev_df,
            'test': test_df
        }


    def _extract_gender_from_key(self, key: str) -> str:
        """
        Extract gender from MELD key (e.g., 'meld-dia1_utt0-test').

        Parameters:
            key (str): The key string from which to extract gender.

        Returns:
            str: The gender of the speaker ('Male' or 'Female').
        """
        sample_name = key.split('-')[1]  # e.g., dia1_utt0
        split = key.split('-')[2]  # e.g., test
        dialogue_full_id = sample_name.split('_')[0]  # e.g., dia1
        utterance_id = sample_name.split('_')[1]  # e.g., utt0
        dialog_id = int(dialogue_full_id.replace('dia', ''))
        utt_id = int(utterance_id.replace('utt', ''))
        df = self.utterance_info.get(split)
        speaker = df[
            (df['Dialogue_ID'] == dialog_id) & (df['Utterance_ID'] == utt_id)
        ]['Speaker'].values[0]
        return SPEAKER_INFO.get(speaker, {}).get('gender', 'Unknown')

    def _extract_age_from_key(self, key: str) -> int:
        """
        Extract age from MELD key (e.g., 'meld-dia1_utt0-test'). Age is calculated as: 
        age = season + base_age (Monica's age in season 1 is 30, 29 + 1)

        Parameters:
            key (str): The key string from which to extract age.

        Returns:
            int: The age of the speaker.
        """
        sample_name = key.split('-')[1]  # e.g., dia1_utt0
        split = key.split('-')[2]  # e.g., test
        dialogue_full_id = sample_name.split('_')[0]  # e.g., dia1
        utterance_id = sample_name.split('_')[1]  # e.g., utt0
        dialog_id = int(dialogue_full_id.replace('dia', ''))
        utt_id = int(utterance_id.replace('utt', ''))

        df = self.utterance_info.get(split)
        speaker = df[
            (df['Dialogue_ID'] == dialog_id) & (df['Utterance_ID'] == utt_id)
        ]['Speaker'].values[0]
        season = int(df[
            (df['Dialogue_ID'] == dialog_id) & (df['Utterance_ID'] == utt_id)
        ]['Season'].values[0])
        
        base_age = SPEAKER_INFO.get(speaker, {}).get('age', 29)
        return base_age + season
    
    def extract_sensitive_attributes(self, entry: dict) -> dict:
        """Extract sensitive attributes for MELD."""
        key = entry.get('key', '')
        return {
            'gender': self._extract_gender_from_key(key),
            'age': self._extract_age_from_key(key),
            'language': self.LANGUAGE,
        }

if __name__ == "__main__":
    annotator = MELDFairnessAnnotator()
    annotator.add_sensitive_attributes_to_all_folds()