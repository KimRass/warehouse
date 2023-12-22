# Paper Summary
- [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
## Related Works
- Over the last few years, researchers have demonstrated the benefits of using word embeddings [11] [39] [42], which are trained on unlabeled corpora, to improve performance on a variety of tasks. These approaches, however, mainly transfer word-level information, whereas we aim to capture higher-level semantics.
- The closest line of work to ours involves pre-training a neural network using a language modeling objective and then fine-tuning it on a target task with supervision.
- Previous work proposed learning task specific architectures on top of transferred representations. Such an approach re-introduces a significant amount of task-specific customization and does not use transfer learning for these additional architectural components.
## Methodology
- Semi-supervised learning
    - we explore a semi-supervised approach for language understanding tasks using a combination of unsupervised pre-training and supervised fine-tuning.
    - ***Our goal is to learn a universal representation that transfers with little adaptation to a wide range of tasks. We assume access to a large corpus of unlabeled text and several datasets with manually annotated training examples (target tasks). Our setup does not require these target tasks to be in the same domain as the unlabeled corpus.***
    - Unsupervised pre-training is a special case of semi-supervised learning where the goal is to find a good initialization point instead of modifying the supervised learning objective. ***Subsequent research demonstrated that pre-training acts as a regularization scheme, enabling better generalization in deep neural networks.***
- Unsupervised pre-training
    - Given an unsupervised corpus of tokens $U = {u_{1}, \dots , u_{n}}$, we use a standard language modeling objective to maximize the following likelihood:
    - Equation 1
    $$L_{1}(\mathcal{U}) = \sum_{i}\log P(u_{i} \mid u_{i − k}, \ldots , u_{i - 1}; Θ)$$
    - where $k$ is the size of the context window, and the conditional probability $P$ is modeled using a neural network with parameters $Θ$.
- Supervised fine-tuning
    - After training the model with the objective in Eq. 1, we adapt the parameters to the supervised target task. We assume a labeled dataset $\mathcal{C}$, where each instance consists of a sequence of input tokens, $x_{1}, \ldots, x_{m}$, along with a label $y$. The inputs are passed through our pre-trained model to obtain the final transformer block’s activation $h^{m}_{l}$, ***which is then fed into an added linear output layer with parameters*** $W_{y}$ ***to predict*** $y$***:***
    - Equation 3
    $$P(y \mid x_{1}, \ldots, x_{m}) = softmax(h^{m}_{l} W_{y})$$
    - This gives us the following objective to maximize:
    - Equation 4
    $$L_{2}(\mathcal{C}) = \sum_{(x,y)}\log P(y \mid x_{1}, \ldots, x_{m})$$
    - ***We additionally found that including language modeling as an auxiliary objective to the fine-tuning helped learning by (a) improving generalization of the supervised model, and (b) accelerating convergence.*** Specifically, we optimize the following objective (with weight $λ$):
    - Equation 5
    $$L_{3}(\mathcal{C}) = L_{2}(\mathcal{C}) + \lambda ∗ L_{1}(\mathcal{C})$$
    - ***Overall, the only extra parameters we require during fine-tuning are*** $W_{y}$***, and embeddings for delimiter tokens.***
## Architecture
- Model specifications Our model largely follows the original Transformer work [62]. ***We trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states.***
- Residual, embedding, and attention dropouts with a rate of 0.1 for regularization.
- ***We also employed a modified version of L2 regularization proposed in [37], with*** $w = 0.01$ ***on all non bias or gain weights.***
- For the activation function, we used the Gaussian Error Linear Unit (GELU).
- ***We used learned position embeddings instead of the sinusoidal version proposed in the original work.***
- Equation 2
$$h_{0} = UW_{e} + W_{p}$$
$$h_{i} = transformer\textunderscore block(h_{i - 1})\ \ \ \ \forall i \in [1, n]$$
$$P(u) = softmax(h_{n}W^{T}_{e})$$
- (Comment: 논문 속 수식의 $l$를 $i$로 변경했습니다.)
- ***where*** $U = (u_{-k}, \ldots , u_{−1})$ ***is the context vector of tokens,*** $n$ ***is the number of layers,*** $W_{e}$ ***is the token embedding matrix, and*** $W_{p}$ ***is the position embedding matrix.***
## Training
- Tokenization
    - We used a bytepair encoding (BPE) vocabulary with 40,000 merges [53].
- We employ a two-stage training procedure. First, we use a language modeling objective on the unlabeled data to learn the initial parameters of a neural network model. Subsequently, we adapt these parameters to a target task using the corresponding supervised objective.
### Pre-training
- ***We used the Adam optimization scheme with a max learning rate of 2.5e-4. The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.***
- ***We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens.*** Since layernorm is used extensively throughout the model, a simple weight initialization of $\mathcal{N}(0, 0.02)$ was sufficient.
- ***We add dropout to the classifier with a rate of 0.1.***
### Fine-tunning
- ***For most tasks, we use a learning rate of 6.25e-5 and a batchsize of 32.*** Our model fine-tunes quickly and 3 epochs of training was sufficient for most cases.
### Finue-tune
- Figure 1. Architecture for fune-tunning
    - <img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-27_at_12.41.44_PM.png" width="800">
    - For some tasks, like text classification, we can directly fine-tune our model as described above. Certain other tasks, like question answering or textual entailment, have structured inputs such as ordered sentence pairs, or triplets of document, question, and answers. ***Since our pre-trained model was trained on contiguous sequences of text, we require some modifications to apply it to these tasks. We use a traversal-style approach, where we convert structured inputs into an ordered sequence that our pre-trained model can process. These input transformations allow us to avoid making extensive changes to the architecture across tasks. All transformations include adding randomly initialized start and end tokens (`<s>`, `<e>`).***
    - ***For entailment tasks, we concatenate the premise*** $p$ ***and hypothesis*** $h$ ***token sequences, with a delimiter token (`$`) in between.***
    - ***For similarity tasks, there is no inherent ordering of the two sentences being compared. To reflect this, we modify the input sequence to contain both possible sentence orderings (with a delimiter in between) and process each independently to produce two sequence representations*** $h^{m}_{l}$ ***which are added element-wise before being fed into the linear output layer.***
    - ***For question answering and commonsense reasoning, we are given a context document*** $z$***, a question*** $q$***, and a set of possible answers*** $\{a_{k}\}$***. We concatenate the document context and question with each possible answer, adding a delimiter token in between to get*** $[z; q; \$ ; a_{k}]$***. Each of these sequences are processed independently with our model and then normalized via a softmax layer to produce an output distribution over possible answers.***
    - Fine-tuning details Unless specified, we reuse the hyperparameter settings from unsupervised pre-training.
- ***We use a linear learning rate decay schedule with warmup over 0.2% of training.*** $\lambda$ ***was set to 0.5.***
## Experiments
- We’d like to better understand why language model pre-training of transformers is effective. ***A hypothesis is that the underlying generative model learns to perform many of the tasks we evaluate on in order to improve its language modeling capability and that the more structured attentional memory of the transformer assists in transfer compared to LSTMs***.
### Transfer Learning
- Figure 2
    - <img src="https://i.imgur.com/z2jLu08.png" width="700">
    - (left) Effect of transferring increasing number of layers from the pre-trained language model on RACE and MultiNLI.
    - (right) Plot showing the evolution of zero-shot performance on different tasks as a function of LM pre-training updates. Performance per task is normalized between a random guess baseline and the current state-of-the-art with a single model.
    - ***We observe the performance of these heuristics is stable and steadily increases over training suggesting that generative pre-training supports the learning of a wide variety of task relevant functionality.***
- Table 5
    - <img src="https://i.imgur.com/1lz0AWO.png" width="700">
    - We examine the performance of our method without the auxiliary LM objective during fine-tuning. ***We observe that the auxiliary objective helps on the NLI tasks and QQP. Overall, the trend suggests that larger datasets benefit from the auxiliary objective but smaller datasets do not.*** (Comment: "Transformer w/ aux LM"과 "Transformer w/o aux LM"을 비교해보면 "QQP", "MNLI", "QNLI", "RTE"에 있어서는 "Transformer w/ aux LM"가, 나머지 Task에 있어서는 "Transformer w/o aux LM"가 성능이 더 뛰어나 것을 볼 수 있습니다.)
    - We also compare with our transformer architecture directly trained on supervised target tasks, without pre-training. We observe that the lack of pre-training hurts performance across all the tasks, resulting in a 14.8 decrease compared to our full model. (Comment: From avearge score 74.7 for "Transformer w/ aux LM" to 59.9 for "Transformer w/o pre-training")


## References
- [37] [Fixing Weight Decay Regularization in Adam](https://openreview.net/pdf?id=rk6qdGgCZ)
- [42] [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- [53] [Neural Machine Translation of Rare Words with Subword Unit](https://arxiv.org/pdf/1508.07909.pdf)
- [62] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
