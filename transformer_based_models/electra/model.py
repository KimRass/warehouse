# References
    # https://huggingface.co/docs/transformers/glossary#token-type-ids
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/electra/modeling_electra.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# from bert.model import BERT
from bert.model import (
    TokenEmbedding,
    SegmentEmbedding,
    PositionEmbedding,
    TransformerBlock,
    _get_pad_mask,
    MaskedLanguageModelHead,
    # NextSentencePredictionHead
)
from bert.tokenize import prepare_bert_tokenizer
from bert.masked_language_model import MaskedLanguageModel

DROP_PROB = 0.1
VOCAB_SIZE = 30_522


class ReplacedTokenDetectionHead(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.cls_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.cls_proj(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers,
        hidden_dim,
        mlp_dim,
        embed_dim,
        n_heads,
        pad_idx=0,
        drop_prob=DROP_PROB,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        self.dropout_p = drop_prob

        self.token_embed = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, pad_idx=pad_idx)
        self.seg_embed = SegmentEmbedding(embed_dim=embed_dim, pad_idx=pad_idx)
        self.pos_embed = PositionEmbedding(embed_dim=embed_dim)
        # "We add linear layers to the generator to project the embeddings into generator-hidden-sized representations."
        # (Comment: ELECTRA-Small의 경우 Generator와 Discriminator 모두 `embed_dim`과 `hidden_dim`이 다릅니다.)
        if hidden_dim != embed_dim:
            self.embed_proj = nn.Linear(embed_dim, hidden_dim)

        self.dropout = nn.Dropout(drop_prob)

        self.tf_block = TransformerBlock(
            n_layers=n_layers, n_heads=n_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim, drop_prob=drop_prob
        )

        self.mlm_head = MaskedLanguageModelHead(vocab_size=vocab_size, hidden_dim=hidden_dim)

        # The 'input' and 'output' token embeddings of the generator are always tied as in BERT."
        self.mlm_head.cls_proj.weight.data = self.token_embed.weight.data # 차원이 맞지 않음!

    def forward(self, seq, seg_ids):
        x = self.token_embed(seq)
        x = self.pos_embed(x)
        x += self.seg_embed(seg_ids)
        x = self.dropout(x)
        if self.hidden_dim != self.embed_dim:
            x = self.embed_proj(x)

        pad_mask = _get_pad_mask(seq=seq, pad_idx=self.pad_idx)
        x = self.tf_block(x, self_attn_mask=pad_mask)
        x = self.mlm_head(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers,
        hidden_dim,
        mlp_dim,
        embed_dim,
        n_heads,
        pad_idx=0,
        drop_prob=DROP_PROB,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        self.dropout_p = drop_prob

        self.token_embed = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, pad_idx=pad_idx)
        self.seg_embed = SegmentEmbedding(embed_dim=embed_dim, pad_idx=pad_idx)
        self.pos_embed = PositionEmbedding(embed_dim=embed_dim)
        # "We add linear layers to the generator to project the embeddings into generator-hidden-sized representations."
        # (Comment: ELECTRA-Small의 경우 Generator와 Discriminator 모두 `embed_dim`과 `hidden_dim`이 다릅니다.)
        if hidden_dim != embed_dim:
            self.embed_proj = nn.Linear(embed_dim, hidden_dim)

        self.dropout = nn.Dropout(drop_prob)

        self.tf_block = TransformerBlock(n_layers=n_layers, n_heads=n_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim)

        self.rtd_head = ReplacedTokenDetectionHead(hidden_dim=hidden_dim)

    def forward(self, seq, seg_ids):
        x = self.token_embed(seq)
        x = self.pos_embed(x)
        x += self.seg_embed(seg_ids)
        x = self.dropout(x)
        if self.hidden_dim != self.embed_dim:
            x = self.embed_proj(x)

        pad_mask = _get_pad_mask(seq=seq, pad_idx=self.pad_idx)
        x = self.tf_block(x, self_attn_mask=pad_mask)
        x = self.rtd_head(x)
        return x


class ELECTRA(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers=12,
        gen_hidden_dim=64,
        disc_hidden_dim=256,
        gen_n_heads=1,
        disc_n_heads=4,
        gen_mlp_dim=256,
        disc_mlp_dim=1024,
        embed_dim=128,
        select_prob=0.15,
        drop_prob=DROP_PROB,
        pad_idx=0
    ):
        super().__init__()

        # vocab_size = VOCAB_SIZE
        # n_layers=12
        # disc_hidden_dim=256
        # gen_hidden_dim=64
        # disc_n_heads=4
        # gen_n_heads=1
        # disc_mlp_dim=1024
        # gen_mlp_dim=256
        # embed_dim=128
        # drop_prob=DROP_PROB
        # pad_idx=0

        mlm = MaskedLanguageModel(
            vocab_size=VOCAB_SIZE,
            mask_id=mask_id,
            no_mask_token_ids=[cls_id, sep_id, mask_id, pad_id],
            select_prob=select_prob, # "Typically 15% of the tokens are masked out."
            mask_prob=1,
            randomize_prob=0,
        )

        # "We speculate that having too strong of a generator may pose a too-challenging task for the discriminator, preventing it from learning as effectively. In particular, the discriminator may have to use many of its parameters modeling the generator rather than the actual data distribution."
        gen = Generator(
            vocab_size=vocab_size,
            n_layers=n_layers,
            hidden_dim=gen_hidden_dim,
            mlp_dim=gen_mlp_dim,
            embed_dim=embed_dim,
            n_heads=gen_n_heads,
            drop_prob=drop_prob,
            pad_idx=pad_idx,
            weight_sharing=True,
        )
        disc = Discriminator(
            vocab_size=vocab_size,
            n_layers=n_layers,
            hidden_dim=disc_hidden_dim,
            mlp_dim=disc_mlp_dim,
            embed_dim=embed_dim,
            n_heads=disc_n_heads,
            drop_prob=drop_prob,
            pad_idx=pad_idx,
        )
        self.tie_weights(gen=self.gen, disc=self.disc)


    def tie_weights(self, gen, disc):
        """
        Weight sharing
        """
        # "We propose improving the efficiency of the pre-training by sharing weights
        # between the generator and discriminator. We found it to be more efficient to have a small generator,
        # in which case we only share the embeddings (both the token and positional embeddings)
        # of the generator and discriminator."
        gen.token_embed = disc.token_embed
        gen.pos_embed = disc.pos_embed


    def forward():
        masked_token_ids, gt_token_ids = mlm(token_ids)
        corrupted_token_ids = gen(masked_token_ids)
        pred_token_ids = disc(corrupted_token_ids)
        return pred_token_ids


class ELECTRASMall(ELECTRA):
    def __init__(self, vocab_size, select_prob=0.15):
        super().__init__(
            vocab_size=vocab_size,
            n_layers=12,
            gen_hidden_dim=64,
            disc_hidden_dim=256,
            gen_n_heads=1,
            disc_n_heads=4,
            gen_mlp_dim=256,
            disc_mlp_dim=1024,
            embed_dim=128,
            select_prob=select_prob,
        )


class ELECTRABase(ELECTRA):
    def __init__(self, vocab_size, select_prob=0.15):
        super().__init__(
            vocab_size=vocab_size,
            n_layers=12,
            gen_hidden_dim=256,
            disc_hidden_dim=768,
            gen_n_heads=4,
            disc_n_heads=12,
            gen_mlp_dim=1024,
            disc_mlp_dim=3072,
            embed_dim=768,
            select_prob=select_prob,
        )


class ELECTRALarge(ELECTRA):
    def __init__(self, vocab_size, select_prob=0.25):
        super().__init__(
            vocab_size=vocab_size,
            n_layers=24,
            gen_hidden_dim=256,
            disc_hidden_dim=1024,
            gen_n_heads=4,
            disc_n_heads=16,
            gen_mlp_dim=1024,
            disc_mlp_dim=4096,
            embed_dim=1024,
            select_prob=select_prob,
        )


vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
cls_id = tokenizer.token_to_id("[CLS]")
sep_id = tokenizer.token_to_id("[SEP]")
mask_id = tokenizer.token_to_id("[MASK]")
pad_id = tokenizer.token_to_id("[PAD]")

