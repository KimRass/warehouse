- Reference: https://github.com/codertimo/BERT-pytorch

import torch
import torch.nn as nn
import random


class SegmentEmbedding(nn.Embedding):
    def __init__(self, dim=512):
        # `num_embeddings`가 왜 2가 아니라 3이지?
        # super().__init__(3, dim, padding_idx=0)
        super().__init__(2, dim, padding_idx=0)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, dim=512):
        super().__init__(vocab_size, dim, padding_idx=0)


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, dim=dim)
        # self.position = PositionEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(dim)
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, sequence, segment_label):
        # x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        x = self.token(sequence) + self.segment(segment_label)
        return self.dropout(x)


if __name__ == "__main__":
    torch.manual_seed(33)

    dim = 512
    vocab_size = 30_000
    segment = SegmentEmbedding(dim)
    token = TokenEmbedding(vocab_size=vocab_size, dim=dim)
    
    bert_embedding = BERTEmbedding(vocab_size=vocab_size, dim=dim)
    sequence = torch.randint(low=0, high=vocab_size - 1, size=(8,))
    sent1_len = random.randint(0, len(sequence) - 1)
    segment_label = torch.as_tensor([0] * sent1_len + [1] * (len(sequence) - sent1_len))
    bert_embedding(sequence=sequence, segment_label=segment_label)