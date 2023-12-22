# References
    # https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/vocab.py

# "Google never open-sourced its implementation of the training algorithm of WordPiece, so what follows
# is the best guess based on the published literature."

from transformers import AutoTokenizer
from collections import defaultdict
from tqdm.auto import tqdm
import json
import re
from pathlib import Path

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")

# "Since it identifies subwords by adding a prefix (like `"##"` for BERT), each word is initially split
# by adding that prefix to all the characters inside the word. For instance, `'word'` gets split like; `'w ##o ##r ##d'`
# Thus, the initial alphabet contains all the characters present at the beginning of a word
# and the characters present inside a word preceded by the WordPiece prefix."

def _lowercase(text):
    text = text.lower()
    return text


def _preprocess(text):
    text = _lowercase(text)
    return text


def _pretokenize(text):
    pretokens = list()
    for i in re.split(pattern=r"[ ]+", string=text):
        for j in re.split(pattern=r"""([ !"#$%&'()*+,-./:;<=>?@\[\\\]^_`{\|}~]+)""", string=i):
            if j:
                pretokens.append(j)
    return pretokens


def _get_pretoken_frequencies(corpus):
    print("Computing frequencies of pretokens...")
    freqs = defaultdict(int)
    for text in tqdm(corpus):
        text = _preprocess(text)
        pretokens = _pretokenize(text)
        for pretoken in pretokens:
            freqs[pretoken] += 1
    print(f"""Number of pretokens: {len(freqs):,}""")
    return freqs


def _build_base_vocabulary(pretokens):
    print("Building base vocabulary...")
    base_vocab = list()
    for pretoken in pretokens:
        if pretoken[0] not in base_vocab:
            base_vocab.append(pretoken[0])
        for char in pretoken[1:]:
            if f"##{char}" not in base_vocab:
                base_vocab.append(f"##{char}")
    base_vocab.sort()
    base_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + base_vocab

    base_vocab = {char: i for i, char in enumerate(base_vocab)}
    print(f"""Size of base vocabulary: {len(base_vocab):,}""")
    return base_vocab


def _split_pretokens(pretokens):
    splits = {
        pretoken: [char if id_ == 0 else f"##{char}" for id_, char in enumerate(pretoken)]
        for pretoken in pretokens
    }
    return splits


def _merge_pair(pair, splits):
    for pretoken in splits:
        split = splits[pretoken]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == pair[0] and split[i + 1] == pair[1]:
                merge = pair[0] + pair[1][2:] if pair[1].startswith("##") else pair[0] + pair[1]
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[pretoken] = split
    return splits


def _compute_pair_scores(freqs, splits):
    char_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for token, freq in freqs.items():
        split = splits[token]
        if len(split) == 1:
            char_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            char_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        char_freqs[split[-1]] += freq

    pair_scores = {
        # Score : $\text{Frequency of pair} / (text{Frequency of first element of pair}
        # \times text{Frequency of second element of pair})$
        pair: freq / (char_freqs[pair[0]] * char_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return pair_scores


def build_or_load_vocab(corpus, vocab_size, save_path):
    if not Path(save_path).exists():
        freqs = _get_pretoken_frequencies(corpus)
        splits = _split_pretokens(pretokens=freqs.keys())
        vocab = _build_base_vocabulary(pretokens=freqs.keys())
        len(vocab.keys())
        if len(vocab) == vocab_size:
            return vocab
        elif len(vocab) > vocab_size:
            vocab = {k: v for i, (k, v) in enumerate(vocab.items(), start=1) if i <= vocab_size}
            return vocab

        with tqdm(total=vocab_size - len(vocab)) as pbar:
            while len(vocab) < vocab_size:
                pair_scores = _compute_pair_scores(freqs=freqs, splits=splits)
                if not pair_scores:
                    break
                best_pair = ("", "")
                max_score = None
                for pair, score in pair_scores.items():
                    if (max_score is None) or (score > max_score):
                        best_pair = pair
                        max_score = score            
                splits = _merge_pair(pair=best_pair, splits=splits)
                new_token = (
                    best_pair[0] + best_pair[1][2:]
                    if best_pair[1].startswith("##")
                    else best_pair[0] + best_pair[1]
                )
                vocab[new_token] = len(vocab)

                pbar.update(1)

        with open(save_path, mode="w") as f:
            json.dump(vocab, f)
        print(f"""Completed building vocabulary! Vocabulary size is {len(vocab):,}.""")

    with open(save_path, mode="r") as f:
        vocab = json.load(f)
    return vocab


def _separate_into_tokens(pretoken, vocab):
    tokens = list()
    while len(pretoken) > 0:
        i = len(pretoken)
        while i > 0 and pretoken[: i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(pretoken[: i])

        pretoken = pretoken[i:]
        if len(pretoken) > 0:
            pretoken = f"##{pretoken}"
    return tokens


def tokenize(text, vocab):
    # text="Hello!"
    # pretokens = TOKENIZER._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    # encoded = [_separate_into_tokens(word=pretoken, vocab=vocab) for pretoken, _ in pretokens]
    pretokens = _pretokenize(text)
    encoded = [_separate_into_tokens(pretoken, vocab=vocab) for pretoken in pretokens]
    return sum(encoded, [])


def encode(text, vocab):
    return [vocab[i] for i in tokenize(text, vocab=vocab)]


def tokens_to_string(tokens):
    text = ""
    for token in tokens:
        if token[: 2] == "##":
            text += token[2:]
        else:
            text += " "
            text += token
    text = text[1:]
    text = re.sub(pattern=r"\[CLS\]|\[SEP\]", repl="", string=text)
    return text


if __name__ == "__main__":
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    vocab = build_or_load_vocab(
        corpus=corpus,
        vocab_size=120,
        save_path="./vocab.json"
    )
