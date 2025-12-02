import json
from abc import ABC, abstractmethod
from pathlib import Path

class FairnessAnnotator(ABC):
    """Base class for adding sensitive attributes to emotion datasets."""
    
    def __init__(self, dataset_name: str, num_folds: int):
        self.dataset_name = dataset_name
        self.num_folds = num_folds
        self.base_dir = f"EmoBox/data/{dataset_name}"
    
    @abstractmethod
    def extract_sensitive_attributes(self, entry: dict) -> dict:
        """
        Extract sensitive attributes from a single entry.
        Returns:
            Dictionary with sensitive attributes (e.g., {'gender': 'Female', 'language': 'English'})
        """
        pass
    
    def add_sensitive_attributes_to_jsonl(self, input_jsonl_path: str):
        """Add sensitive attributes to existing JSONL file."""
        entries = []
        with open(input_jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        for entry in entries:
            entry['sensitive_attr'] = self.extract_sensitive_attributes(entry)
        
        with open(input_jsonl_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
    
    def add_sensitive_attributes_to_folder(self, folder_path: str, pattern: str = "*.jsonl"):
        """Add sensitive attributes to all JSONL files in a folder."""
        folder = Path(folder_path)
        jsonl_files = list(folder.glob(pattern))
        
        if not jsonl_files:
            raise FileNotFoundError(f"No files found in {folder_path} with pattern {pattern}")
        
        print(f"Found {len(jsonl_files)} JSONL files")
        
        for jsonl_file in jsonl_files:
            print(f"Processing: {jsonl_file}")
            self.add_sensitive_attributes_to_jsonl(str(jsonl_file))
    
    def add_sensitive_attributes_to_all_folds(self, base_dir: str = None):
        """Add sensitive attributes to all fold files in dataset."""
        base_dir = base_dir or self.base_dir
        
        for fold in range(1, self.num_folds + 1):
            fold_dir = Path(base_dir) / f"fold_{fold}"
            
            if not fold_dir.exists():
                print(f"Fold directory not found: {fold_dir}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing Fold {fold}")
            print(f"{'='*60}")
            
            self.add_sensitive_attributes_to_folder(str(fold_dir))