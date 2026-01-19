import os
import json
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import librosa
import numpy as np
import logging
import unicodedata
# import torchaudio
SAMPLING_RATE=16000
logger = logging.getLogger(__name__)



"""
[{
	key: "Ses01M_impro01_F000"
	dataset: "iemocap"
	wav: "Session1/sentences/wav/Ses01M_impro01/Ses01M_impro01_F000.wav"
    type: "raw" # raw, feature
	sample_rate: 16000
	length: 3.2
	task: "category" # category, valence, arousal
	emo: "hap"
	}, 
    {...}
]
"""
def check_exists(data, data_dir, logger):
    new_data = []
    for instance in data:
        audio_path = unicodedata.normalize('NFC', instance['wav'])
        if os.path.exists(audio_path):
            new_data.append(instance)
    print(f'load in {len(data)} samples, only {len(new_data)} exists in data dir {data_dir}')        
    return new_data

def replace_label(data, label_map, logger):
    new_data = []
    for instance in data:
        emotion = instance['emo']
        label = label_map[emotion]
        instance['emo'] = label
        new_data.append(instance)
    return new_data

def filter_by_language(data, language):
    filtered_data = []
    for instance in data:
        sensitive_attr = instance.get('sensitive_attr', {})
        instance_lang = sensitive_attr.get('language', None)
        if instance_lang == language:
            filtered_data.append(instance)
    print(f'Filtered from {len(data)} to {len(filtered_data)} samples for language: {language}')
    return filtered_data

def prepare_data_from_jsonl(
    dataset,
    meta_data_dir,
    label_map,
    fold=1,
    split_ratio=[80, 20],
    seed=12,
    language=None,
):
    # setting seeds for reproducible code.
    random.seed(seed)

    
    # find train/valid/test metadata files
    train_data_path = os.path.join(meta_data_dir, dataset, f'fold_{fold}', f'{dataset}_train_fold_{fold}.jsonl')
    test_data_path = os.path.join(meta_data_dir, dataset, f'fold_{fold}', f'{dataset}_test_fold_{fold}.jsonl')
    valid_data_path = os.path.join(meta_data_dir, dataset, f'fold_{fold}', f'{dataset}_valid_fold_{fold}.jsonl')

    # check existance
    assert os.path.exists(train_data_path), f'train data path {train_data_path} does not exist!'
    assert os.path.exists(test_data_path), f'test data path {test_data_path} does not exist!'
    official_valid = False
    if os.path.exists(valid_data_path):
        print(f'using official valid data in {valid_data_path}')
        official_valid = True
    else:
        print(f'since there is no official valid data, use random split for train valid split, with a ratio of {split_ratio}')    

    # load in train & test data
    train_data = []
    test_data = []
    valid_data = []
    with open(train_data_path) as f:
        for line in f:
            train_data.append(json.loads(line.strip()))
    with open(test_data_path) as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    if official_valid:
        with open(valid_data_path) as f:
            for line in f:
                valid_data.append(json.loads(line.strip()))
            
    train_data = check_exists(train_data, meta_data_dir, logger)
    test_data = check_exists(test_data, meta_data_dir, logger)
    if official_valid:
        valid_data = check_exists(valid_data, meta_data_dir, logger)
    else:
        train_data, valid_data = split_sets(train_data, split_ratio)

    if language is not None:
        train_data = filter_by_language(train_data, language)
        valid_data = filter_by_language(valid_data, language)
        test_data = filter_by_language(test_data, language)
        
    num_train_data = len(train_data)
    num_valid_data = len(valid_data)
    num_test_samples = len(test_data)
    print(f'Num. training samples {num_train_data}')
    print(f'Num. valid samples {num_valid_data}')
    print(f'Num. test samples {num_test_samples}')

    
    print(f'Using label_map {label_map}')
    train_data = replace_label(train_data, label_map, logger)
    valid_data = replace_label(valid_data, label_map, logger)
    test_data = replace_label(test_data, label_map, logger)
    
    return train_data, valid_data, test_data

def split_sets(train_data, split_ratio):
    num_train_data = len(train_data)
    num_train_nodev_samples = int(num_train_data * split_ratio[0])

    sample_idx = np.arange(num_train_data)
    random.shuffle(sample_idx)
    train_nodev_data = [ train_data[idx] for idx in sample_idx[:num_train_nodev_samples]]
    valid_data = [train_data[idx] for idx in sample_idx[num_train_nodev_samples:]]
    
    return train_data, valid_data

# Modified to load audio with soundfile instead of torchaudio
def read_wav(data):
    wav_path = data['wav']
    is_webm = wav_path.lower().endswith('.webm')
    wav_path = unicodedata.normalize('NFC', wav_path)
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"{wav_path} does not exist.")
    channel = data['channel']
    dur = float(data['length'])
    if 'start_time' in data and 'end_time' in data:
        start_time = data['start_time']
        end_time = data['end_time']
    else:
        start_time = None
        end_time = None
    
    is_webm = wav_path.lower().endswith('.webm')
    is_mp4 = wav_path.lower().endswith('.mp4')
    is_m4a = wav_path.lower().endswith('.m4a')
    use_librosa = is_webm or is_mp4 or is_m4a
    
    if start_time is not None and end_time is not None:
        # sample_rate = torchaudio.info(wav_path).sample_rate
        # num_frames = int(end_time * sample_rate) - frame_offset
        # wav, sr = torchaudio.load(wav_path, frame_offset=frame_offset, num_frames=num_frames)

        if use_librosa:
            duration = end_time - start_time
            wav, sr = librosa.load(
                wav_path, 
                sr=None, 
                offset=start_time, 
                duration=duration,
                mono=False
            )
        else:
            sample_rate = sf.info(wav_path).samplerate
            frame_offset = int(start_time * sample_rate)
            num_frames = int(end_time * sample_rate) - frame_offset
            wav, sr = sf.read(wav_path, start=frame_offset, frames=num_frames,dtype='float32')

        # duration = end_time - start_time
        # wav, sr = librosa.load(
        #     wav_path, 
        #     sr=SAMPLING_RATE, 
        #     offset=start_time, 
        #     duration=duration,
        #     mono=True
        # )
    else:
        if use_librosa:
            wav, sr = librosa.load(wav_path, sr=None, mono=False)
        else:
            # wav, sr = torchaudio.load(wav_path)
            wav, sr = sf.read(wav_path, dtype='float32')
            # wav, sr = librosa.load(wav_path, sr=SAMPLING_RATE, mono=True)

    if wav.ndim > 1:
        # librosa returns (n_channels, n_samples), soundfile returns (n_samples, n_channels)
        # Check which format we have by comparing dimensions
        if wav.shape[0] < wav.shape[1]:
            # librosa format: (channels, samples) - average along channel axis
            wav = wav.mean(axis=0)
        else:
            # soundfile format: (samples, channels) - average along channel axis
            wav = wav.mean(axis=-1)

    if sr != SAMPLING_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLING_RATE)
        
    return wav.astype(np.float32)

class EmoDataset(Dataset):
    def __init__(self, dataset, data_dir, meta_data_dir, fold=1, split="train", language=None):
        super().__init__()
        self.name = dataset
        self.data_dir = data_dir
        self.label_map = json.load(
            open(os.path.join(meta_data_dir, dataset, 'label_map.json'))
        )
        train_data, valid_data, test_data = prepare_data_from_jsonl(
            dataset, meta_data_dir, self.label_map, fold=fold, language=language
        )
        if split == 'train':
            self.data_list = train_data
        elif split == 'valid':
            self.data_list = valid_data
        elif split == 'test':
            self.data_list = test_data
        else:
            raise Exception(f'does not support split {split}') 
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        key = data["key"]  
        audio = read_wav(data)
        label = data['emo']
        sensitive_attr = data.get('sensitive_attr', {})
        return{
            "key": key,
            "audio": audio,
            "label": label,
            **sensitive_attr,
            # other meta data can be added here
        }

if __name__ == "__main__":
    # test code
    meta_data_dir = 'EmoBox/data'
    data_dir = './data/'
    # dataset = 'emozionalmente'
    dataset = 'meld'
    fold = 1
    test_set = EmoDataset(dataset, data_dir, meta_data_dir, fold=fold, split='test')
    print(f'Num. training samples: {len(test_set)}')
    for idx in range(3):
        sample = test_set[idx]
        print(sample['key'], sample['audio'].shape, sample['label'])