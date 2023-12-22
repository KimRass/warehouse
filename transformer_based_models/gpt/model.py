# References:
    # https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/learned_positional_embedding.py
    # https://paul-hyun.github.io/gpt-01/?fbclid=IwAR3jaAPdcWBIkShNDr-NIXE5JCfw-UvoQ2h000r5qnSBj8kjrY4ax1jDeM8
    # https://gaussian37.github.io/dl-pytorch-lr_scheduler/

# Model specifications Our model largely follows the original transformer work [62]. We trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states. We used the Adam optimization scheme [27] with a max learning rate of 2.5e-4. The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.
# Since layernorm [2] is used extensively throughout the model, a simple weight initialization of N(0; 0:02) was sufficient.
# We use the ftfy library2 to clean the raw text in BooksCorpus, standardize some punctuation and whitespace, and use the spaCy tokenizer.

# We also employed a modified version of L2 regularization proposed in [37], with w = 0:01 on all non bias or gain weights.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.model import ResidualConnection, MultiHeadAttention, PositionwiseFeedForward, _get_pad_mask
from bert.model import TokenEmbedding

torch.set_printoptions(precision=3, edgeitems=4, linewidth=sys.maxsize)

# "We used Residual, embedding, and attention dropouts with a rate of 0.1 for regularization."
# We add dropout to the classifier with a rate of 0.1. ??
DROP_PROB = 0.1
VOCAB_SIZE = 40_000 # "We used a bytepair encoding (BPE) vocabulary with 40,000 merges."???


# "We used learned position embeddings instead of the sinusoidal version proposed in the original work."
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim, pad_id=0):
        super().__init__()

        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pad_id = pad_id

        self.embed = nn.Embedding(num_embeddings=max_len + 1, embedding_dim=embed_dim, padding_idx=pad_id)

    def forward(self, x):
        not_pad = (x != self.pad_id)
        x = torch.cumsum(not_pad, dim=1) * not_pad
        x = self.embed(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, attn_drop_prob=DROP_PROB, resid_drop_prob=DROP_PROB):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.attn_drop_prob = attn_drop_prob
        self.resid_drop_prob = resid_drop_prob

        self.self_attn = MultiHeadAttention(dim=dim, n_heads=n_heads, drop_prob=attn_drop_prob)
        self.attn_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)
        # "For the activation function, we used the Gaussian Error Linear Unit (GELU)."
        self.feed_forward = PositionwiseFeedForward(dim=dim, mlp_dim=mlp_dim, activ="gelu")
        self.ff_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)

    def forward(self, x, self_attn_mask):
        x = self.attn_resid_conn(x=x, sublayer=lambda x: self.self_attn(q=x, k=x, v=x, mask=self_attn_mask))
        x = self.ff_resid_conn(x=x, sublayer=self.feed_forward)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_heads, mlp_dim, attn_drop_prob, resid_drop_prob):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim

        self.dec_stack = nn.ModuleList([
            TransformerLayer(
                dim=hidden_dim,
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                attn_drop_prob=attn_drop_prob, # "Attention dropout"
                resid_drop_prob=resid_drop_prob, # "Residual dropout"
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, self_attn_mask):
        for dec_layer in self.dec_stack:
            x = dec_layer(x, self_attn_mask=self_attn_mask)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers=12,
        hidden_dim=768,
        n_heads=12,
        max_len=512,
        pad_id=0,
        embed_drop_prob=DROP_PROB,
        attn_drop_prob=DROP_PROB,
        resid_drop_prob=DROP_PROB,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.max_len = max_len
        self.pad_id = pad_id
        self.embed_drop_prob = embed_drop_prob
        self.attn_drop_prob = attn_drop_prob
        self.resid_drop_prob = resid_drop_prob

        self.token_embed = TokenEmbedding(vocab_size=vocab_size, embed_dim=hidden_dim, pad_id=pad_id)
        self.pos_embed = LearnedPositionalEmbedding(max_len=max_len, embed_dim=hidden_dim, pad_id=pad_id)

        self.embed_drop = nn.Dropout(embed_drop_prob) # "Embedding dropout"

        self.tf_block = TransformerBlock(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            mlp_dim=hidden_dim * 4,
            attn_drop_prob=attn_drop_prob,
            resid_drop_prob=resid_drop_prob,
        )
    
    def forward(self, seq):
        x = self.token_embed(seq) + self.pos_embed(seq)
        x = self.embed_drop(x)

        pad_mask = _get_pad_mask(seq=seq, pad_id=self.pad_id)
        x = self.tf_block(x, self_attn_mask=pad_mask)
        return x


if __name__ == "__main__":
    # "We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens."
    # BATCH_SIZE = 64
    BATCH_SIZE = 4
    MAX_LEN = 512

    seq = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
    gpt = GPT(vocab_size=VOCAB_SIZE)
    logit = gpt(seq)
    print(logit.shape)
