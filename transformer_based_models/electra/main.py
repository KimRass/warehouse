# References
    # https://github.com/lucidrains/electra-pytorch/blob/master/pretraining/openwebtext/pretrain.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert.model import BERT

def tie_weights(gen, disc):
    """
    Weight sharing
    """
    gen.token_embed = disc.token_embed
    gen.pos_embed = disc.pos_embed
    # gen.electra.embeddings.token_type_embeddings = disc.electra.embeddings.token_type_embeddings

VOCAB_SIZE = 30_522
# ELECTRA-Small: `n_layers=12, hidden_dim=256, n_heads=4`
# ELECTRA-Base: `n_layers=12, hidden_dim=768, n_heads=12`
# ELECTRA-Large: `n_layers=24, hidden_dim=1024, n_heads=16`
disc = BERT(vocab_size=VOCAB_SIZE, n_layers=12, hidden_dim=256, n_heads=4)

MASK_PROB = 0.15 # For ELECTRA-Small and ELECTRA-Base
MASK_PROB = 0.25 # For ELECTRA-Large
DISC_WEIGHT = 50