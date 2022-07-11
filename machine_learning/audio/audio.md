# Dataset
## KsponSpeech (Korean Spontaneous Speech Corpus for Automatic Speech Recognition)
- Reference: https://www.mdpi.com/2076-3417/10/19/6936

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
```python
import librosa

# When computing an STFT, you compute the FFT for a number of short segments. These segments have the length `n_fft`. Usually these segments overlap (in order to avoid information loss), so the distance between two segments is often not `n_fft`, but something like `n_fft/2`. The name for this distance is `hop_length`. It is also defined in samples.
frame_leng = 20
frame_shift = 10
n_fft = int(round(sr * 0.001 * frame_leng))
hop_leng = int(round(sr * 0.001 * frame_shift))
n_mels = int(len(y)/1000)
S = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=n_fft, hop_length=hop_leng, n_mels=n_mels
)
S_db = librosa.amplitude_to_db(S, ref=np.max)
```

# Tasks
## STT (Speech-To-Text) (= ASR (Automatic Speech Recognition))
## TTS (Text-To-Speech) (= Speech Synthesis)

# wave2vec

# Tacotron2
- References: https://github.com/Rayhane-mamah/Tacotron-2, https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS
- ![tacotron2_architecture](https://camo.githubusercontent.com/d6c3e238b30a49a31c947dd0c5b344c452b53ab5eb735dc79675b67c92a2cf96/68747470733a2f2f707265766965772e6962622e636f2f625538734c532f5461636f74726f6e5f325f4172636869746563747572652e706e67)