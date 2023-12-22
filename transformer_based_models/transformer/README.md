# Paper Reading
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
## 3.4 Embeddings and Softmax
- In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation. In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$.