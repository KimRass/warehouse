import numpy as np
from datasets import Dataset, load_dataset, load_metric
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm.auto import tqdm
import re
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import json
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)


def read_audio(audio_path):
    ext = Path(audio_path).suffix
    if ext in [".wav", ".flac"]:
        y, sr = sf.read(audio_path, dtype="int16")
        if y.ndim == 2:
            print("This audio file is stereo!")
        y = y.astype(np.int32)
    elif ext == ".pcm":
        y = np.memmap(audio_path, dtype="h", mode="r")
    return y, sr


def normalize_signal(signal):
    return signal / (2 ** 15)


def create_pebble_dataset(data_dir):
    data_dir = Path(data_dir)

    ls_audio_path = []
    ls_signal_norm = []
    ls_transcript = []
    for audio_path in tqdm(sorted(data_dir.glob("*/*.wav"))):
        signal, _ = read_audio(audio_path)
        signal_norm = normalize_signal(signal)
        
        txt_path = Path(str(audio_path).replace("wav", "txt"))
        with open(txt_path, mode="r") as f:
                transcript = f.read()
        ls_signal_norm.append(signal_norm)
        ls_transcript.append(transcript)
        ls_audio_path.append(str(audio_path))

    dic_for_ds = {"path": ls_audio_path, "audio": ls_signal_norm, "transcript": ls_transcript}
    ds = Dataset.from_dict(dic_for_ds)
    return ds


def load_feature_extractor():
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    #     "padding_side": "right",
    return feature_extractor


def load_tokenizer():
    tokenizer = Wav2Vec2CTCTokenizer(
        # "./vocab.json"
        "/Users/jongbeom.kim/Desktop/workspace/data_science/machine_learning/audio/vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
        do_lower_case=False,
        word_delimiter_token="|"
    )
    return tokenizer


def load_processor(feature_extractor, tokenizer):
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    return processor
