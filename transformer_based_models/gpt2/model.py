# An additional layer normalization was added after the final self-attention block. A modified initialization which accounts
# "A modified initialization which accounts for the accumulation on the residual path with model depth is used."
# "We scale the weights of residual layers at initialization by a factor of $1 / \sqrt{N}$ where $N$ is the number of residual layers."

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert.model import ResidualConnection

# "Layer normalization was moved to the input of each sub-block."

MAX_LEN = 1024 # "We also increase the context size from 512 to 1024 tokens
VOCAB_SIZE = 50_257 # "The vocabulary is expanded to 50,257."
BATCH_SIZE = 512 # and a larger batchsize of 512 is used."

# "The smallest model is equivalent to the original GPT."
class SmallestGPT2(GPT):
    n_layers=12, hidden_dim=768


# "The second smallest equivalent to the largest model from BERT."
class SecondSmallestGPT2(GPT):
    n_layers=24, hidden_dim=1024


class SecondLargestGPT2(GPT):
    n_layers=36, hidden_dim=1280


# "Our largest model, which we call 'GPT-2', has over an order of magnitude more parameters than GPT."
class GPT2(GPT):
    n_layers=48, hidden_dim=1600
