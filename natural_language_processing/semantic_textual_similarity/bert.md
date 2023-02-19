# Paper Summary
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- **The pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks without substantial task-specific architecture modifications.**
- Feature-based approach
- Fine-tuning approach
    - **The fine-tuning approach, such as the GPT, introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all pre-trained parameters.**
## Related Works
- In OpenAI GPT, the authors use a left-to-right architecture, where every token can only at- tend to previous tokens in the self-attention layers of the Transformer. Such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying fine-tuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.
## Train
- There are two steps in our framework: pre-training and fine-tuning.
### Pre-training
- During pre-training, the model is trained on unlabeled data over different pre-training tasks.
#### Masked Language Model (MLM)
- BERT alleviates the previously mentioned unidirectionality constraint by using a "masked lan- guage model" (MLM) pre-training objective.
- **The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer.**
#### Next Sentence Prediction (NSP)
- In addition to the masked language model, we also use a "next sentence prediction" task that jointly pre-trains text-pair representations.
### Fine-tuning
- **For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters.**