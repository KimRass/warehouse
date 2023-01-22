# BERT (Bidirectional Encoder Representations from Transformer)
- Reference: https://www.youtube.com/watch?v=30SvdoA6ApE, https://www.youtube.com/watch?v=IwtexRHoWG0, https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

# Architecture
- BERT-Base: Number of layers 12, Hidden size 768, Number of self attention heads 12, Parameters 110M
- BERT-Large: Number of layers 24, Hidden size 1,024, Number of self attention heads 16, Parameters 340M
- In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary.
- As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. Therefore it is considered bidirectional, though it would be more accurate to say that it’s non-directional. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word).
## Inputs
- ![bert_input](https://miro.medium.com/max/640/1*iJqlhZz-g6ZQJ53-rE9VvA.png)
- ![bert_input2](https://miro.medium.com/max/1400/1*Nvha0X2B-63AmlwMGiHtjQ@2x.png)
- '[CLS]': The first token of every sequence
- '[SEP]': Special token separating two sentences
- Number of parameters:
  - Token embeddings: 30k (Vocabulary size) x 768 = 23,040,000
  - Segment embeddings: 2개의 문장을 Input으로 넣을 경우 768 x 2 = 1,536
  - Position embeddings: 768 x 512 (Length of sequence) = 393,216

# Pre-training
- If the i-th token is chosen to be masked, it is replace by the '[MASK]' token 80% of the time (e.g., 'My dog is [MASK]'), a random token 10% of the time (e.g., 'My dog is apple'), and unchanged 10% of the time (e.g., 'My dog is hairy').
## Masked LM (MLM)
- ![mlm](https://miro.medium.com/max/1400/0*ViwaI3Vvbnd-CJSQ.png)
- 15% of the words in each sequence are replaced with a '[MASK]' token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence.
## Next Sentence Prediction (NSP)
- 50% of the time sentence B is the actual next sentence that follows sentence A (IsNext)
- 50% of the time it is a random sentence from the corpus (NotNext)

# Fine-Tunning