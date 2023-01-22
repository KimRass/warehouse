# CTC (Connectionsist Temporal Classification)
- Reference: https://distill.pub/2017/ctc/, https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7
- The output of the NN is a matrix containing character-probabilities for each time-step (horizontal position).
- A path encodes text in the following way: each character of a text can be repeated an arbitrary number of times. Further, an arbitrary number of CTC blanks (non-character, not to be confused with a white-space character, denoted as "-" in this article) can be inserted between characters. In the case of repeated characters (e.g. "pizza"), at least one blank must be placed in between these repeated characters on the path (e.g. 'piz-za').
- As you see, there may be more than one path corresponding to a text. When we are interested in the probability of a text, we have to sum over the probabilities of all corresponding paths. The probability of a single path is the product of the character-probabilities on this path.
# Beam Search in CTC
- Beam search decoding iteratively creates text candidates (beams) and scores them.
- The beam width (BW) specifies the number of beams to keep.