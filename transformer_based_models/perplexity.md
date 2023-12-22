# Perplexity
- We are also often interested in the probability that our model assigns to a full sentence $W$ made of the sequence of words $(w_{1},w_{2}, \dots, w_{N})$.
$$P(W) = P(w_{1},w_{2}, \dots, w_{N})$$
- ***We want our model to assign high probabilities to sentences that are real and syntactically correct, and low probabilities to fake, incorrect, or highly infrequent sentences.***
- ***Assuming our dataset is made of sentences that are in fact real and correct, this means that the best model will be the one that assigns the highest probability to the test set. Intuitively, if a model assigns a high probability to the test set, it means that it is not surprised to see it (it’s not perplexed by it), which means that it has a good understanding of how the language works.***
## Evaluation
- Extrinsic evaluation:
    - This involves evaluating the models by employing them in an actual task (such as machine translation) and looking at their final loss/accuracy. This is the best option as it’s the only way to tangibly see how different models affect the task we’re interested in. However, it can be computationally expensive and slow as it requires training a full system.
- Intrinsic evaluation:
    - This involves finding some metric to evaluate the language model itself, not taking into account the specific tasks it’s going to be used for. While intrinsic evaluation is not as "good" as extrinsic evaluation as a final metric, it’s a useful way of quickly comparing models. Perplexity is an intrinsic evaluation method.
## Normalized Inverse Probability of The Test Set
$$PPL(W) = \frac{1}{P(w_{1},w_{2}, \dots, w_{N})^{\frac{1}{N}}}$$
- Since we’re taking the inverse probability, a lower perplexity indicates a better model.
- ***In this case W is the test set. It contains the sequence of words of all sentences one after the other, including the start-of-sentence and end-of-sentence tokens, `"<SOS>"` and `"<EOS>"`. For example, a test set with two sentences would look like this: W = (`"<SOS>"`, This, is, the, first, sentence, .,`"<EOS>"`, `"<SOS>"`, This, is, the, second, one, ., `"<EOS>"`). $N$ is the count of all tokens in our test set, including `"<SOS>"`, `"<EOS>"` and punctuation. In the example above N = 16. If we want, we can also calculate the perplexity of a single sentence, in which case W would simply be that one sentence.***
## Branching Factor
- ***We can interpret perplexity as the weighted branching factor. If we have a perplexity of 100, it means that whenever the model is trying to guess the next word it is as confused as if it had to pick between 100 words.***
## References
- [1] [Perplexity in Language Models](https://towardsdatascience.com/perplexity-in-language-models-87a196019a94)