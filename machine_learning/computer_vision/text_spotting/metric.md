## Error Rate (CER (Character Error Rate), WER (Word Error Rate))
- Reference: https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510
- Edit distance
  - Reference: https://en.wikipedia.org/wiki/Edit_distance
  - Edit distance is a way of quantifying how dissimilar two strings (e.g., words) are to one another by counting the minimum number of operations required to transform one string into the other.
  - Levenshtein distance
    - Allows deletion, insertion and substitution.
    - It is the minimum number of single-character (or word) edits (i.e., insertions, deletions, or substitutions) required to change one word (or sentence) into another. The more different the two text sequences are, the higher the number of edits needed, and thus the larger the Levenshtein distance.
- The usual way of evaluating prediction output is with the accuracy metric, where we indicate a match (1) or a no match (0). However, this does not provide enough granularity to assess OCR performance effectively. We should instead use error rates to determine the extent to which the OCR transcribed text and ground truth text (i.e., reference text labeled manually) differ from each other. A common intuition is to see how many characters were misspelled. While this is correct, the actual error rate calculation is more complex than that. This is because the OCR output can have a different length from the ground truth text.
- The question now is, how do you measure the extent of errors between two text sequences? This is where Levenshtein distance enters the picture.
- CER calculation is based on the concept of Levenshtein distance, where we count the minimum number of character-level operations required to transform the ground truth text (aka reference text) into the OCR output.
- CER = (S + D + I) / N
  - N: Number of characters in reference text (aka ground truth)
- Normalized CER = (S + D + I) / (S + D + I + C)
  - C: Number of correct characters
```python
import fastwer

cer = fastwer.score_sent(output, ref, char_level=True)
wer = fastwer.score_sent(output, ref, char_level=False)
```