import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import re

from bert.wordpiece import tokens_to_string
from transformer.model import Transformer, _get_pad_mask, _get_subsequent_info_mask


if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/self_based_models/bert/vocab.json"
    with open(vocab_path, mode="r") as f:
        vocab = json.load(f)
    pad_id = vocab["[PAD]"]
    cls_id = vocab["[CLS]"]
    unk_id = vocab["[UNK]"]
    sep_id = vocab["[SEP]"]
    mask_id = vocab["[MASK]"]

    BATCH_SIZE = 16
    SEQ_LEN = 30
    VOCAB_SIZE = 1000
    src_pad_idx = 0
    trg_pad_idx = 0
    transformer = Transformer(
        src_vocab_size=VOCAB_SIZE,
        trg_vocab_size=VOCAB_SIZE,
        src_seq_len=SEQ_LEN,
        trg_seq_len=SEQ_LEN,
        src_pad_idx=pad_id,
        trg_pad_idx=trg_pad_idx
    )

    src_seq = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    gen_seq = torch.zeros(size=(BATCH_SIZE, SEQ_LEN), dtype=torch.int64)


def vocab_ids_to_strings(seq, id_to_token):
    arr = seq.detach().cpu().numpy()
    ls = arr.tolist()
    ls = [list(map(lambda x: id_to_token.get(x, unk_id), i)) for i in ls]
    return ls


def infer(model, src_seq):
    model = transformer

    model.eval()
    src_pad_mask = _get_pad_mask(seq=src_seq, pad_idx=model.src_pad_idx)
    # trg_pad_mask = _get_pad_mask(seq=gen_seq, pad_idx=model.trg_pad_idx)
    # trg_subseq_mask = _get_subsequent_info_mask(src_seq_len=model.src_seq_len, trg_seq_len=model.trg_seq_len)

    enc = model.enc
    dec = model.dec

    enc_output = enc(src_seq, self_attn_mask=src_pad_mask)
    is_finished = torch.zeros(size=(BATCH_SIZE,), dtype=torch.bool)
    with torch.no_grad():
        gen_seq[:, 0] = cls_id
        for idx in range(1, SEQ_LEN):
            dec_logits = dec(
                gen_seq,
                enc_output=enc_output
                # self_attn_mask=trg_pad_mask,
                # enc_dec_mask=(trg_pad_mask | trg_subseq_mask)
            )
            gen_tokens = torch.argmax(dec_logits[:, idx, :], dim=1)
            gen_seq[:, idx] = gen_tokens

            # Stop inferencing
            is_finished = torch.logical_or(is_finished, (gen_tokens == sep_id))
            if is_finished.sum().item() == BATCH_SIZE:
                break

    id_to_token = {v:k for k, v in vocab.items()}
    tokens = vocab_ids_to_strings(seq=gen_seq, id_to_token=id_to_token)
    sents = [tokens_to_string(i) for i in tokens]
    return sents
