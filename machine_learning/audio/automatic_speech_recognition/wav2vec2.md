# Wave2Vec 2.0
- Similar, to BERT's masked language modeling, the model learns contextualized speech representations by randomly masking feature vectors before passing them to a transformer network.
## XLSR (= XLSR-Wav2Vec) (Cross-lingual Speech Representations)
- Paper: https://arxiv.org/abs/2006.13979
- Reference: https://paperswithcode.com/method/xlsr
- XLSR is a multilingual speech recognition model built on wav2vec 2.0 which is trained by solving a contrastive task over masked latent speech representations and jointly learns a quantization of the latents shared across languages. *The model is fine-tuned on labeled data and experiments show that cross-lingual pretraining significantly outperforms monolingual pretraining.* The model learns to share discrete tokens across languages, creating bridges across languages.
- Before fine-tuning a pretrained checkpoint of an ASR model, it is crucial to verify that the sampling rate of the data that was used to pretrain the model matches the sampling rate of the dataset used to fine-tune the model.
- XLSR-Wav2Vec2 was pretrained on the audio data of Babel, Multilingual LibriSpeech (MLS), and Common Voice. Most of those datasets were sampled at 16kHz.
```python
# `Wav2Vec2FeatureExtractor`: Processes the speech signal to the model's input format, e.g. a feature vector
# `Wav2Vec2CTCTokenizer`: Processes the model's output format to text.
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

# `feature_size`: In the case of Wav2Vec2, the feature size is `1` because the model was trained on the raw speech signal.
# `padding_value`: For batched inference, shorter inputs need to be padded with a specific value.
# `do_normalize`: Whether the input should be zero-mean-unit-variance normalized or not. Usually, speech models perform better when normalizing the input.
# `return_attention_mask`: Whether the model should make use of an attention_mask for batched inference. In general, XLSR-Wav2Vec2 models should always make use of the attention_mask.
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

# `padding`
  # `True`, `"longest"`: Pad to the longest sequence in the batch.
  # `"max_length"`: `max_length` is needed.
  # `False`, `"do_not_path"`
# `max_length`
  # `None`: It will default to the maximum length the model can accept.
batch = self.processor.pad()

DataCollatorWithPadding
# `pad_to_multiple_of`: If set will pad the sequence to a multiple of the provided value.
```
```python
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer
)
# `pad()`: When used in normal mode, this method forwards all its arguments to `Wav2Vec2FeatureExtractor`’s `pad()` and returns its output. If used in the context `processor.as_target_processor()` this method forwards all its arguments to `Wav2Vec2CTCTokenizer`’s `pad()`.
# `batch_decode()`: This method forwards all its arguments to PreTrainedTokenizer’s batch_decode(). Please refer to the docstring of this method for more information.

# we do not want to group tokens when computing the metrics
# `group_tokens`: To allow models to become independent of the speaker rate, in CTC, consecutive tokens that are identical are simply grouped as a single token. However, the encoded `input_ids` should not be grouped when decoding since they don't correspond to the predicted tokens of the model, which is why the `group_tokens=False` parameter has to be passed. If we wouldn"t pass this parameter a word like "hello" would incorrectly be encoded, and decoded as "helo".
# The blank token allows the model to predict a word, such as "hello" by forcing it to insert the blank token between the two l"s. A CTC-conform prediction of "hello" of our model would be [PAD] [PAD] "h" "e" "e" "l" "l" [PAD] "l" "o" "o" [PAD].
processor.batch_decode()
```
```python
# Returns the list of decoded sentences.
tokenizer.batch_decode()
```