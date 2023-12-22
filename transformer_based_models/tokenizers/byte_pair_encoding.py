# References
    # https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt

from collections import defaultdict, OrderedDict
import re
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def _split_on_punctuation(text):
    ls = list()
    start = 0
    for m in re.finditer(pattern=r"""([!"#$%&'()*+,-./:;<=>?@\[\\\]^_`{\|}~]+)""", string=text):
        end = m.start()
        trg = text[start: end]
        if "Ġ" == trg:
            ls.append(trg + text[end: m.end()])
        else:
            ls.append(trg)
            ls.append(text[end: m.end()])
        start = m.end()
    trg = text[start:]
    if trg:
        ls.append(trg)
    if ls:
        return ls
    else:
        return [text]


def pretokenize(text):
    text = text.replace(" ", "Ġ") + "Ġ"

    ls = list()
    string = ""
    for id_ in range(len(text) - 1):
        char1 = text[id_]
        if char1 == "Ġ":
            if string:
                ls.extend(_split_on_punctuation(string))
            string = ""
            char2 = text[id_ + 1]
            if char2 == "Ġ":
                ls.append(char1)
            else:
                string += char1
        else:
            string += char1
    ls.extend(_split_on_punctuation(string))
    return ls


def get_pretoken_frequencies(corpus):
    freqs = defaultdict(int)
    for text in tqdm(corpus):
        # "A tokenizer cannot be trained on raw text alone. Instead, we first need to split the texts
        # into small entities, like words. That’s where the pre-tokenization step comes in.
        # It will split on whitespace and punctuation as well, but it will keep the spaces
        # and replace them with a `'Ġ'` symbol, enabling it to recover the original spaces if we decode the tokens.
        # Unlike the BERT tokenizer, byte pair encoding does not ignore the double space."
        pretokens = pretokenize(text)
        for pretoken in pretokens:
            freqs[pretoken] += 1
    return freqs


def get_character_level_vocabulary(pretokens):
    # "For real-world cases, that base vocabulary will contain all the ASCII characters, at the very least,
    # and probably some Unicode characters as well.
    vocab = list()
    for pretoken in pretokens:
        for char in pretoken:
            if char not in vocab:
                vocab.append(char)
    vocab.sort()

    vocab = ["<|endoftext|>"] + vocab.copy()
    return vocab


def _compute_pair_frequencies(freqs):
    pair_freqs = defaultdict(int)
    for word, freq in freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def _merge_pair(a, b, splits):
    for word in freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


def build_vocab(splits, vocab_size):
    vocab = get_character_level_vocabulary(pretokens=freqs.keys())
    # "We also add the special tokens used by the model at the beginning of that vocabulary. In the case of GPT-2,
    # the only special token is `'<|endoftext|>'`."

    merges = OrderedDict()
    while len(vocab) < vocab_size:
        pair_freqs = _compute_pair_frequencies(splits)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits = _merge_pair(*best_pair, splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])
    # "The vocabulary is composed of the special token, the initial alphabet, and all the results of the merges."

    # "Using 🤗 Tokenizers library on the same corpus won’t result in the exact same vocabulary. This is because
    # when there is a choice of the most frequent pair, we selected the first one encountered, while the 🤗 Tokenizers library
    # selects the first one based on its inner IDs."
    return vocab, merges


def tokenize(text, merges):
    # "To tokenize a new text, we pre-tokenize it, split it, then apply all the merge rules learned"
    pretokenized_text = pretokenize(text)
    splits = [[l for l in word] for word in pretokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split
    return sum(splits, [])


if __name__ == "__main__":
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    freqs = get_pretoken_frequencies(corpus)
    # "BPE training starts by computing the unique set of words used in the corpus
    # (after the normalization and pre-tokenization steps are completed)"
    
    # "We now need to split each word into individual characters, to be able to start training"
    splits = {word: [c for c in word] for word in freqs.keys()}

    vocab, merges = build_vocab(splits=splits, vocab_size=50)
    tokenize("This is not a token.", merges=merges)
