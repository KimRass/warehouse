# Zero-Shot Learning
- Reference: https://joeddav.github.io/blog/2020/05/29/ZSL.html
## Projection from Sentence BERT to Wor2Vec
- Take the top K most frequent words V in the vocabulary of a word2vec model.
## Topic Classification as Natural Language Inference
- Paper: https://arxiv.org/abs/1909.00161
- As a quick review, natural language inference (NLI) considers two sentences: a "premise" and a "hypothesis". The task is to determine whether the hypothesis is true (entailment) or false (contradiction) given the premise.
- When using transformer architectures like BERT, NLI datasets are typically modeled via sequence-pair classification. That is, we feed both the premise and the hypothesis through the model together as distinct segments and learn a classification head predicting one of ("contradiction", "neutral", "entailment").
- ***The idea is to take the sequence we're interested in labeling as the "premise" and to turn each candidate label into a "hypothesis." If the NLI model predicts that the premise "entails" the hypothesis, we take the label to be true.***
- Classes to use: conversation, art, beauty, fashion, commerce, education, law, literature, religion, politics, society, economy, financial, sports, science, medicine, technology, travel
- Classes not to use: legal, patent, liberal arts, transport, computer, shopping, engineering, business, customer service, marketing, social media, information technology