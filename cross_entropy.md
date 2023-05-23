- https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e

# PyTorch
- References:
    - https://junstar92.tistory.com/118
    - https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- `F.nll_loss(F.log_softmax())`와 `F.cross_entropy()`는 서로 동일합니다.
## `F.cross_entropy()`
- `input`: Predicted unnormalized logits.
- `target`: Ground truth class indices or class probabilitie.
