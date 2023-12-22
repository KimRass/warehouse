# References
    # https://nn.labml.ai/transformers/rope/index.html

import sys
import torch
from torch import nn

from transformer.model import MultiHeadAttention

torch.set_printoptions(edgeitems=16, linewidth=sys.maxsize, sci_mode=True)


# "Rotary encoding organizes the $d$ features as $d/2$ pairs. Each pair can be considered
# a coordinate in a 2D plane, and the encoding will rotate it by an angle depending on the position of the token."

# We pair feature $i$ with feature $i + 2/d$. So for position $m$
# If $i \in \{1, 2, ... d / 2\}$
# $$x^{(i)}_{m}$$
# is transformed to
# $$x^{(i)}_{m}\cos{m\theta_{i}} + (-x^{(i + d / 2)}_{m})\sin{m\theta_{i}}$$
# and otherwise transformed to
# $$x^{(i + d / 2)}_{m}\cos{m\theta_{i}} + x^{(i)}_{m}\sin{m\theta_{i}}$$

# $$\langle \text{RoPE}(x^{(1)}_{m}​, x^{(2)}_{m}​, m), \text{RoPE}(x^{(1)}_{n}​, x^{(2)}_{n}​, n) \rangle =\
# \langle \text{RoPE}(x^{(1)}_{m}​, x^{(2)}_{m}​, m - n), \text{RoPE}(x^{(1)}_{n}​, x^{(2)}_{n}​, 0) \rangle$$
# "This shows that for dot-production attention the rotary encodings gives relative attention."
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, base=10_000):
        super().__init__()

        self.head_dim = head_dim
        self.base = base

    def _get_theta(self, i):
        # $\Theta = \{\theta_{i} = 10000^{-2(i - 1)/d}, i \in [1, 2, \ldots d/2]\}$
        return self.base ** (-2 * (i - 1) / self.head_dim)

    # `x` is the tensor at the head of a key or a query with shape (`batch_size`, `n_heads`, `seq_len`, `head_dim`)
    def forward(self, x):
        _, _, seq_len, _ = x.shape

        pos = torch.arange(seq_len, dtype=x.dtype) # $m$
        i = torch.arange(1, self.head_dim // 2 + 1).repeat(2) # $i$ # 1, 2, ..., d / 2| 1, 2, ... d / 2
        theta = self._get_theta(i) # $\theta_{i}$
        v = torch.einsum("p,t->pt", pos, theta) # $m\theta_{i}$

        self.cos_mtheta = torch.cos(v) # $\cos{m\theta_{i}}$
        self.sin_mtheta = torch.sin(v) # $\sin{m\theta_{i}}$

        #             1,             2, ...,    d // 2 - 1|    d // 2, d // 2 + 1, ...,      d - 1,      d
        # -(1 + d / 2), -(2 + d / 2), ...,      -(d - 1)|      -(d),          1, ..., d / 2 - 1, d / 2
        pair = torch.cat([-x[..., self.head_dim // 2:], x[..., : self.head_dim // 2]], dim=3)
        x = x * self.cos_mtheta + pair * self.sin_mtheta
        return x


class RoPEMultiHeadAttention(MultiHeadAttention):
    def __init__(self, dim, n_heads, drop_prob, rope_prob=0.5):
        super().__init__(dim=dim, n_heads=n_heads, drop_prob=drop_prob)

        self.rope_prob = rope_prob

        # self.rope_dim = int(self.head_dim * self.rope_prob)
        self.rope_dim = dim // n_heads
        self.q_rope, self.k_rope = (
            RotaryPositionalEmbedding(head_dim=self.rope_dim),
            RotaryPositionalEmbedding(head_dim=self.rope_dim)
        )

    def _get_attention_score(self, q, k):
        attn_score = torch.einsum("bnid,bnjd->bnij", self.q_rope(q), self.k_rope(k))
        return attn_score


if __name__ == "__main__":
    BATCH_SIZE = 16
    N_HEADS = 8
    SEQ_LEN = 30
    DIM = 96
    rope = RotaryPositionalEmbedding(dim=DIM)
    x = torch.randn((BATCH_SIZE, N_HEADS, SEQ_LEN, DIM))
    out = rope(x)
    print(x.shape, out.shape)


    DIM = 512
    N_HEADS = 8
    DROP_PROB = 0.1
    rope_mha = RoPEMultiHeadAttention(dim=DIM, n_heads=N_HEADS, drop_prob=DROP_PROB)
    x = torch.randn((BATCH_SIZE, SEQ_LEN, DIM))
    out = rope_mha(q=x, k=x, v=x)
    out.shape