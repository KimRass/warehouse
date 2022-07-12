import numpy as np
from datasets import load_dataset, load_metric, Audio
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor

common_voice_train = load_dataset("common_voice", "ja", split="train+validation")
common_voice_test = load_dataset("common_voice", "ja", split="test")

common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])


def create_vocab_csv(ls_transcript, vocab_size, save_dir):
    print("Creating 'vocab.csv'...")

    label_list = list()
    label_freq = list()
    for transcript in ls_transcript:
        for char in transcript:
            if char not in label_list:
                label_list.append(char)
                label_freq.append(1)
            else:
                label_freq[label_list.index(char)] += 1

    label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
    # label = {"idx": [0, 1, 2], "char": ["<pad>", "<sos>", "<eos>"], "freq": [0, 0, 0]}
    label = {"idx": [0], "char": ["<pad>"], "freq": [0]}

    for idxx, (char, freq) in enumerate(zip(label_list, label_freq)):
        # label["idx"].append(idxx + 3)
        label["idx"].append(idxx + 1)
        label["char"].append(char)
        label["freq"].append(freq)

    label["idx"] = label["idx"][:vocab_size]
    label["char"] = label["char"][:vocab_size]
    label["freq"] = label["freq"][:vocab_size]

    label_df = pd.DataFrame(label)
    label_df.to_csv(save_dir / "vocab.csv", encoding="utf-8", index=False)

    print("Completed creating 'vocab.csv'!")


def load_feature_extractor():
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    # {
    #     "do_normalize": true,
    #     "feature_size": 1,
    #     "padding_side": "right",
    #     "padding_value": 0.0,
    #     "return_attention_mask": true,
    #     "sampling_rate": 16000
    # }
    return feature_extractor


def load_tokenizer():
    # tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token=" ")
    tokenizer = Wav2Vec2CTCTokenizer(
        "/Users/jongbeom.kim/Desktop/workspace/data_science/machine_learning/audio/vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
        do_lower_case=False
        word_delimiter_token="|"
    )
    # {"unk_token": "[UNK]", "bos_token": "<s>", "eos_token": "</s>", "pad_token": "[PAD]", "do_lower_case": false, "word_delimiter_token": "|"}
    return tokenizer


def load_processor(feature_extractor, tokenizer):
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    return processor


# def prepare_dataset(batch):
#     audio = batch["audio"]

#     # batched output is "un-batched"
#     batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
#     with processor.as_target_processor():
#         batch["labels"] = processor(batch["sentence"]).input_ids
#     return batch


def main():
    feature_extractor = load_feature_extractor()
    tokenizer = load_tokenizer()
    processor = load_processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    audio_path = "/Users/jongbeom.kim/Documents/ksponspeech/data/KsponSpeech_01/KsponSpeech_0123/KsponSpeech_122988.pcm"
    signal = np.memmap(audio_path, dtype="h", mode="r") / 2 ** 15
    signal

    sentence = "그래서 지호랑 계단 올라와서 막 위에 운동하는 기구 있대요. 그서 그걸로 운동 할려구요."
    sentence = sentence.replace(" ", "|").replace(".", "")
    with processor.as_target_processor():
        tokens = processor(sentence).input_ids
    print(tokens)