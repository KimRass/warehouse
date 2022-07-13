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


def bracket_filter(sentence, preprocess_mode="phonetic"):
    new_sentence = ""
    if preprocess_mode == "phonetic":
        flag = False
        for char in sentence:
            if char == "(" and flag is False:
                flag = True
                continue
            if char == "(" and flag is True:
                flag = False
                continue
            if char != ")" and flag is False:
                new_sentence += char
    elif preprocess_mode == "spelling":
        flag = True
        for char in sentence:
            if char == "(":
                continue
            if char == ")":
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if char != ")" and flag is True:
                new_sentence += char
    else:
        raise ValueError(f"Unsupported mode: {preprocess_mode}")
    return new_sentence


def special_filter(sentence, preprocess_mode="phonetic", replace=None):
    SENTENCE_MARK = ["?", "!", "."]
    NOISE = ["o", "n", "u", "b", "l"]
    EXCEPT = ["/", "+", "*", "-", "@", "$", "^", "&", "[", "]", "=", ":", ";", ","]

    new_sentence = ""
    for idx, char in enumerate(sentence):
        if char not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and char in NOISE and sentence[idx + 1] == "/":
                continue
        if char == "#":
            new_sentence += "샾"
        elif char == "%":
            if preprocess_mode == "phonetic":
                new_sentence += replace
            elif preprocess_mode == "spelling":
                new_sentence += "%"
        elif char not in EXCEPT:
            new_sentence += char

    pattern = re.compile(r"\s\s+")
    new_sentence = re.sub(pattern, " ", new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, preprocess_mode, replace=""):
    sentence=bracket_filter(raw_sentence, preprocess_mode)
    return special_filter(
        sentence=sentence,
        preprocess_mode=preprocess_mode,
        replace=replace
    )


def create_dataset_from_ksponspeech(data_dir):
    data_dir = Path(data_dir)
    
    percent_files = {
            "087797": "퍼센트",
            "215401": "퍼센트",
            "284574": "퍼센트",
            "397184": "퍼센트",
            "501006": "프로",
            "502173": "프로",
            "542363": "프로",
            "581483": "퍼센트"
        }

    ls_audio_path = list()
    ls_signal_norm = list()
    ls_sentence = list()
    for i, audio_path in enumerate(tqdm(sorted(data_dir.glob("*/*/*.pcm")))):
        signal, _ = read_audio(audio_path)
        signal_norm = normalize_signal(signal)
        
        txt_path = Path(str(audio_path).replace("pcm", "txt"))
        with open(txt_path, mode="r", encoding="cp949") as f:
                data_idx = txt_path.name.split("_")[-1]
                sentence = f.read()
                if data_idx in percent_files:
                    sentence = sentence_filter(
                        raw_sentence=sentence,
                        preprocess_mode="phonetic",
                        replace=percent_files[data_idx])
                else:
                    sentence = sentence_filter(
                        raw_sentence=sentence,
                        preprocess_mode="phonetic"
                    )
                sentence = replace_some_characters(sentence)
        ls_signal_norm.append(signal_norm)
        ls_sentence.append(sentence)
        ls_audio_path.append(str(audio_path))
        if i == 100:
            break

    dic_for_ds = {"path": ls_audio_path, "audio": ls_signal_norm, "sentence": ls_sentence}
    ds = Dataset.from_dict(dic_for_ds)
    return ds

ds["audio"][0]
def read_audio(audio_path, sr=16000):
    ext = Path(audio_path).suffix
    if ext in [".wav", ".flac"]:
        y, sr = sf.read(audio_path, dtype="int16")
        y = y.astype(np.int32)
    elif ext == ".pcm":
        y = np.memmap(audio_path, dtype="h", mode="r")
    return y, sr


def normalize_signal(signal):
    return signal / 2 ** 15


# def create_vocab_csv(ls_transcript, vocab_size, save_dir):
#     print("Creating "vocab.csv"...")

#     label_list = list()
#     label_freq = list()
#     for transcript in ls_transcript:
#         for char in transcript:
#             if char not in label_list:
#                 label_list.append(char)
#                 label_freq.append(1)
#             else:
#                 label_freq[label_list.index(char)] += 1

#     label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
#     # label = {"idx": [0, 1, 2], "char": ["<pad>", "<sos>", "<eos>"], "freq": [0, 0, 0]}
#     label = {"idx": [0], "char": ["<pad>"], "freq": [0]}

#     for idxx, (char, freq) in enumerate(zip(label_list, label_freq)):
#         # label["idx"].append(idxx + 3)
#         label["idx"].append(idxx + 1)
#         label["char"].append(char)
#         label["freq"].append(freq)

#     label["idx"] = label["idx"][:vocab_size]
#     label["char"] = label["char"][:vocab_size]
#     label["freq"] = label["freq"][:vocab_size]

#     label_df = pd.DataFrame(label)
#     label_df.to_csv(save_dir / "vocab.csv", encoding="utf-8", index=False)

#     print("Completed creating "vocab.csv"!")


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


def replace_some_characters(sentence: str) -> str:
    return sentence.replace(" ", "|").replace(".", "")


def remove_special_characters(batch):
    chars_to_ignore_regex = "[\,\?\.\!\-\;\:\"\“\%\‘\”\�]"
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).lower()
    # + " "
    return batch


def prepare_dataset(processor, batch):
    batch["input_values"] = processor(
        batch["audio"], sampling_rate=16000
    ).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch


def compute_metrics(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    # To allow models to become independent of the speaker rate, in CTC, consecutive tokens that are identical are simply grouped as a single token. However, the encoded labels should not be grouped when decoding since they don"t correspond to the predicted tokens of the model, which is why the group_tokens=False parameter has to be passed. If we wouldn"t pass this parameter a word like "hello" would incorrectly be encoded, and decoded as "helo".
    # The blank token allows the model to predict a word, such as "hello" by forcing it to insert the blank token between the two l"s. A CTC-conform prediction of "hello" of our model would be [PAD] [PAD] "h" "e" "e" "l" "l" [PAD] "l" "o" "o" [PAD].
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_metric = load_metric("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    feature_extractor = load_feature_extractor()
    tokenizer = load_tokenizer()
    processor = load_processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    ds = create_dataset_from_ksponspeech("/Users/jongbeom.kim/Documents/ksponspeech/data")
    # ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds2 = ds.map(remove_special_characters)
    ds3 = ds2.map(lambda x: prepare_dataset(processor=processor, batch=x), remove_columns=ds2.column_names, num_proc=4)
    
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    
    # The first component of XLSR-Wav2Vec2 consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal. This part of the model has already been sufficiently trained during pretraining and as stated in the paper does not need to be fine-tuned anymore. Thus, we can set the requires_grad to False for all parameters of the feature extraction part.
    model.freeze_feature_extractor()
    # This Saves memory.
    model.gradient_checkpointing_enable()
    
    # Reference: https://huggingface.co/docs/transformers/main/main_classes/trainer#trainingarguments
    # group_by_length makes training more efficient by grouping training samples of similar input length into one batch. This can significantly speed up training time by heavily reducing the overall number of useless padding tokens that are passed through the model
    training_args = TrainingArguments(
        output_dir="./wav2vec2-large-xlsr-ko",
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds3,
        eval_dataset=ds3,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()

common_voice_test = load_dataset("common_voice", "tr", split="test")
common_voice_test[0]["path"]
common_voice_test["path"][0]

type(common_voice_test[0]["audio"]["array"])
type(dic_for_ds["audio"][0])

type(ds["audio"][0])
type(ds[0]["audio"])
common_voice_test[0]["audio"]



percent_files = {
    "087797": "퍼센트",
    "215401": "퍼센트",
    "284574": "퍼센트",
    "397184": "퍼센트",
    "501006": "프로",
    "502173": "프로",
    "542363": "프로",
    "581483": "퍼센트"
}

ls_audio_path = list()
ls_signal_norm = list()
ls_sentence = list()
temp = dict()
for i, audio_path in enumerate(tqdm(sorted(data_dir.glob("*/*/*.pcm")))):
    signal, _ = read_audio(audio_path)
    signal_norm = normalize_signal(signal)
    
    txt_path = Path(str(audio_path).replace("pcm", "txt"))
    with open(txt_path, mode="r", encoding="cp949") as f:
            data_idx = txt_path.name.split("_")[-1]
            sentence = f.read()
            if data_idx in percent_files:
                sentence = sentence_filter(
                    raw_sentence=sentence,
                    preprocess_mode="phonetic",
                    replace=percent_files[data_idx])
            else:
                sentence = sentence_filter(
                    raw_sentence=sentence,
                    preprocess_mode="phonetic"
                )
            sentence = replace_some_characters(sentence)
    ls_signal_norm.append(signal_norm)
    ls_sentence.append(sentence)
    ls_audio_path.append(str(audio_path))
    if i == 100:
        break
    temp
dic_for_ds = {"audio": temp, "sentence": ls_sentence}



arr = np.array([1, 2, 3])
ds = Dataset.from_dict({"audio": {"array": arr}})
ds["audio"]

common_voice_test[0]["audio"]["array"]

dataset = load_dataset("json", data_files="/Users/jongbeom.kim/Desktop/workspace/data_science/machine_learning/audio/vocab.json")

# json_str = json.dumps(dic_for_ds, default=lambda x: x.__dict__, indent=4, ensure_ascii=False)
json_str = json.dumps(dic_for_ds, indent=4)
json_path = "/Users/jongbeom.kim/Desktop/workspace/data_science/machine_learning/audio/ksopnspeech.json"
with open(json_path, mode="w") as f:
    f.write(json_str)