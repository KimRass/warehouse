# Datasets
## KsponSpeech (Korean Spontaneous Speech Corpus for Automatic Speech Recognition)
- Reference: https://www.mdpi.com/2076-3417/10/19/6936
## Common Voice
- Reference: https://commonvoice.mozilla.org/en/datasets

# Audio File Extensions
- Reference: https://en.wikipedia.org/wiki/Audio_file_format
- Uncompressed audio formats: WAV, AIFF, AU raw header-less PCM
  - *WAV and AIFF are designed to store a wide variety of audio formats, lossless and lossy; they just add a small, metadata-containing header before the audio data to declare the format of the audio data, such as LPCM with a particular sample rate, bit depth, endianness and number of channels.* Since WAV and AIFF are widely supported and can store LPCM, they are suitable file formats for storing and archiving an original recording.
- Formats with lossless compression: FLAC, ALAC (".m4a")
- Formats with lossy compression: MP3, AAC
## WAV
- 무손실, 무압축 방식입니다.
## PCM (Pulse Code Modulation)
- Header 없이 Raw data만 저장한 파일입니다.
- 별도로 Audio에 대한 정보를 가지고 있지 않으면 재생할 수 없습니다.
- Reference: https://www.openmyfiles.com/pcm-file/
- PCM file is used to store a raw digital audio file in a computer system. It is a direct representation of (1s and 0s) digital sample values of an audio recording. The file keeps the recording in raw digital audio since high-quality recordings can be lossless. The files can’t use any form of compression to cut out the less important recordings to reduce the size of the file. PCM files stores the audio data samples in a sequentially binary format. Sometimes the audio files can be stored in WAV and AIF file format. The wave WAV file is the most common file used for storing PCM recordings.

# Utterance
- Reference: https://en.wikipedia.org/wiki/Utterance
- In spoken language analysis, an utterance is the smallest unit of speech. *It is a continuous piece of speech beginning and ending with a clear pause. In the case of oral languages, it is generally, but not always, bounded by silence.* Utterances do not exist in written language; only their representations do. They can be represented and delineated in written language in many ways.

# Propagation of Sound
- Reference: https://blog.paperspace.com/introduction-to-audio-analysis-and-synthesis/
- The parts where air is pushed closer together are called compressions, and the parts where it is pushed further apart are called rarefactions. Such waves that traverse space using compressions and rarefactions are called longitudinal waves.

# Fourier Transform
- Reference: https://blog.paperspace.com/introduction-to-audio-analysis-and-synthesis/
- We call this transformation moving from a time-domain representation to a frequency domain representation.
- A discrete Fourier transform is computationally quite heavy to calculate, with a time complexity of the order O(n^2). But there is a faster algorithm called Fast Fourier Transform (or FFT) that performs with a complexity of O(nlogn).
## Short-Time Fourier Transform
- With Fourier transforms, we convert a signal from the time domain into the frequency domain. In doing so, we see how every point in time is interacting with every other for every frequency. Short-time Fourier transforms do so for the neighboring points in time instead of the entire signal. This is done by utilizing a window function that hops with a specific hop length to give us the frequency domain values.
## Spectrogram
- The STFT can provide a rich visual representation for us to analyze, called a spectrogram. A spectrogram is a two-dimensional representation of the square of the STFT, and can give us important visual insight into which parts of a piece of audio sound like a buzz, a hum, a hiss, a click, or a pop, or if there are any gaps.
- Utilizing the STFT matrix directly to plot doesn't give us clear information. A common practice is to convert the amplitude spectrogram into a power spectrogram by squaring the matrix. Following this, converting the power in our spectrogram to decibels against some reference power increases the visibility of our data.
```python
# References: https://librosa.org/doc/main/generated/librosa.stft.html#:~:text=The%20default%20value%2C%20n_fft%3D2048,well%20adapted%20for%20music%20signals., https://stackoverflow.com/questions/63350459/getting-the-frequencies-associated-with-stft-in-librosa
# `n_fft`: In speech processing, the recommended value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz. In any case, we recommend setting `n_fft` to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm.
# For each FFT window, for real signals the spectrum is symmetric so we only consider the positive side of the FFT. So The number of rows in the STFT matrix is `1 + n_fft/2`, with 1 being the DC component. Since we are only looking at the half spectrum, instead of `i` spanning from `0` to `n_fft`, this spans from `0` up to `1 + n_fft / 2` instead as the bins beyond `1 + n_fft / 2` would simply be the reflected version of the half spectrum.
stft = librosa.stft(y_norm, n_fft=n_fft,  hop_length=hop_length)
spectrogram = np.abs(stft)
spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
```
## Mel-Spectrogram
- The Mel Scale
  - Reference: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
  - Studies have shown that humans do not perceive frequencies on a linear scale. We are better at detecting differences in lower frequencies than higher frequencies. For example, we can easily tell the difference between 500 and 1000 Hz, but we will hardly be able to tell a difference between 10,000 and 10,500 Hz, even though the distance between the two pairs are the same.
    ```python
    # Create a Mel filter-bank. This produces a linear transformation matrix to project FFT bins onto Mel-frequency bins.
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    ```
- Reference: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
- The Mel Spectrogram is the result of the following pipeline:
  - Separate to windows: Sample the input with windows of size `n_fft`, making hops of size `hop_length` each time to sample the next window.
  - Compute FFT (Fast Fourier Transform) for each window to transform from time domain to frequency domain.
  - Generate a Mel scale: Take the entire frequency spectrum, and separate it into `n_mels` evenly spaced frequencies. ***And what do we mean by evenly spaced? not by distance on the frequency dimension, but distance as it is heard by the human ear.***
  - Generate Spectrogram: For each window, decompose the magnitude of the signal into its components, corresponding to the frequencies in the mel scale.
