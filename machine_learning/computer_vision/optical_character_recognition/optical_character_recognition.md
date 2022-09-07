# Preprocess
- Reference: https://medium.com/technovators/survey-on-image-preprocessing-techniques-to-improve-ocr-accuracy-616ddb931b76
## Image Binarization
## Deskewing (= Skew Correcting)
## Sharpening
- By increasing the local contrast (i.e) contrast between text and background, each character will be easily distinguishable from the background and makes it easier to recognize the character. Similarly, having sharp borders between characters will be helpful for character segmentation and recognition.
- In most cases, using the global contrast is not a good option because different parts of the image may have different contrasts. Hence, Contrast Limited Adaptive Histogram Equalization (CLAHE) is a very effective pre-processing step to improve the text and background contrast.
## Trapezoidal Distortion Correction
## Correction of 3D perspective distortions
## Denoising (= Noise Removal)
### Blurring or Smoothing
- Gaussian Blur
- Median Blur
- Bilateral Filtering
- Both bilateral and median filter is good at preserving edges but the former one is very slow due to its computational complexity.
## Using AutoEncoders
## Image Despeckling
## ISO Noise Correction