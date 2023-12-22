import torch
import torch.nn as nn

from bert.model import TokenEmbedding, PositionEmbedding, TransformerBlock, _get_pad_mask, BERTBase

DROP_PROB = 0.1


class DistilBERTBase(nn.Module):
    def __init__(
        self,
        teacher,
        vocab_size,
        n_layers=6, # "The number of layers is reduced by a factor of 2." in section 3 of the paper.
        n_heads=12,
        hidden_dim=768,
        mlp_dim=768 * 4,
        pad_idx=0,
        drop_prob=DROP_PROB
    ):
        super().__init__()

        self.teacher = teacher
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.pad_idx = pad_idx

        # "The token-type embeddings are removed."
        self.token_embed = TokenEmbedding(vocab_size=vocab_size, embed_dim=hidden_dim, pad_idx=pad_idx)
        self.pos_embed = PositionEmbedding(embed_dim=hidden_dim)

        self.drop = nn.Dropout(drop_prob)

        self.tf_block = TransformerBlock(
            n_layers=n_layers, n_heads=n_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim, drop_prob=drop_prob
        )
        # The pooler are removed.

    def initialize(self):
        for i in range(self.n_layers):
            self.tf_block.enc_stack[i].self_attn.q_proj.weight.data =\
                self.teacher.tf_block.enc_stack[2 * i].self_attn.q_proj.weight.data
            self.tf_block.enc_stack[i].self_attn.k_proj.weight.data =\
                self.teacher.tf_block.enc_stack[2 * i].self_attn.k_proj.weight.data
            self.tf_block.enc_stack[i].self_attn.v_proj.weight.data =\
                self.teacher.tf_block.enc_stack[2 * i].self_attn.v_proj.weight.data
            self.tf_block.enc_stack[i].self_attn.out_proj.weight.data =\
                self.teacher.tf_block.enc_stack[2 * i].self_attn.out_proj.weight.data
            self.tf_block.enc_stack[i].norm1.weight.data =\
                self.teacher.tf_block.enc_stack[2 * i].norm1.weight.data
            self.tf_block.enc_stack[i].ff.proj1.weight =\
                self.teacher.tf_block.enc_stack[2 * i].ff.proj1.weight
            self.tf_block.enc_stack[i].ff.proj2.weight =\
                self.teacher.tf_block.enc_stack[2 * i].ff.proj2.weight

    def forward(self, seq):
        x = self.token_embed(seq)
        x = self.pos_embed(x)
        x = self.drop(x)

        pad_mask = _get_pad_mask(seq=seq, pad_idx=self.pad_idx)
        x = self.tf_block(x, self_attn_mask=pad_mask)
        return x


if __name__ == "__main__":
    VOCAB_SIZE = 30_522
    teacher = BERTBase(vocab_size=VOCAB_SIZE)
    distil_bert = DistilBERTBase(teacher=teacher, vocab_size=VOCAB_SIZE)
    print(distil_bert.tf_block.enc_stack[0].ff.proj1.weight[0, 1].item())
    distil_bert.initialize()
    print(distil_bert.tf_block.enc_stack[0].ff.proj1.weight[0, 1].item())
