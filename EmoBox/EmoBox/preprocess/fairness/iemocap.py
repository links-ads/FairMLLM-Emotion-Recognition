import os
import json

from pathlib import Path

DATASET_NAME = "iemocap"
NUM_FOLDS = 5
LANGUAGE = "English"

def extract_gender_from_key(key: str) -> str:
    gender_code = key.split('_')[-1][0]
    if gender_code == 'F':
        return 'Female'
    elif gender_code == 'M':
        return 'Male'

def add_sensitive_attributes_to_jsonl(input_jsonl_path: str):
    """
    Add sensitive attributes (gender) to existing JSONL file.
    """
    entries = []
    with open(input_jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    for entry in entries:
        key = entry.get('key', '')
        gender = extract_gender_from_key(key)
        entry['sensitive_attr'] = {
            'gender': gender,
            'language': LANGUAGE,
        }
    
    with open(input_jsonl_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

def add_sensitive_attributes_to_folder(
    folder_path: str,
    pattern: str = "*.jsonl",
):
    """
    Add sensitive attributes to all JSONL files in a folder.
    """
    folder = Path(folder_path)
    jsonl_files = list(folder.glob(pattern))
    
    if not jsonl_files:
        raise FileNotFoundError(f"No *{pattern} files found in {folder_path} with pattern {pattern}")
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing: {jsonl_file}")
        add_sensitive_attributes_to_jsonl(str(jsonl_file))

def add_sensitive_attributes_to_all_folds(base_dir: str = "data/iemocap"):
    """
    Add sensitive attributes to all fold files in IEMOCAP dataset.
    """
    for fold in range(1, NUM_FOLDS + 1):
        fold_dir = os.path.join(base_dir, f"fold_{fold}")
        
        if not os.path.exists(fold_dir):
            print(f"Fold directory not found: {fold_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold}")
        print(f"{'='*60}")
        
        add_sensitive_attributes_to_folder(
            fold_dir,
            pattern="*.jsonl",
        )

if __name__ == "__main__":
    add_sensitive_attributes_to_all_folds(
        base_dir="EmoBox/data/iemocap",
    )