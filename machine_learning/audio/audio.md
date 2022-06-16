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
- human sound perception is not linear, and we are able to differentiate between lower frequencies a lot better than higher frequencies. This is captured by the mel scale.