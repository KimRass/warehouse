# Preprocess
- Reference: https://medium.com/technovators/survey-on-image-preprocessing-techniques-to-improve-ocr-accuracy-616ddb931b76
## Image Binarization
## Deskewing (= Skew Correcting)
## Sharpening
- By increasing the local contrast (i.e) contrast between text and background, each character will be easily distinguishable from the background and makes it easier to recognize the character. Similarly, having sharp borders between characters will be helpful for character segmentation and recognition.
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

# CRAFT (Character Region Awareness For Text detection)
- Paper: https://arxiv.org/abs/1904.01941
- References: https://github.com/clovaai/CRAFT-pytorch, https://github.com/LoveGalaxy/Character-Region-Awareness-for-Text-Detection-, https://github.com/ducanh841988/Kindai-OCR, https://medium.com/@msmapark2/character-region-awareness-for-text-detection-craft-paper-%EB%B6%84%EC%84%9D-da987b32609c
- OCR을 Character-level로 수행합니다.
- Input으로 주어진 이미지의 각 픽셀마다 "Region score"와 "Affinity score"를 예측합니다.
  - Region score: 해당 픽셀이 어떤 문자의 중심일 확률.
  - Affiniy score: 해당 픽셀이 어떤 문자와 다른 어떤 문자의 중심일 확률. 이것을 통해 여러 문자들을 묶어서 하나의 텍스트로 인식할지 여부가 결정됩니다.
- 다음의 세 가지를 예측할 수 있습니다; Character boxes, Word boxes, Polygons
- Inference stage: Character region을 바탕으로 위에서 말한 세 가지를 추론하는 단계
  - Word-level QuadBox (word-level bounding box) Inference
    - Polygon Inference
      - ![polygon_inference](https://miro.medium.com/max/1400/1*_EyygIYQyQPqUk-w-OaKjw.png)
      - "Local maxima along scanning direction" (Blue. "Control points of text polygon"의 후보들) -> "Center line of local maxima" (Yellow) -> "Line of control points" (Red. 문자 기울기 반영) -> 양 끝에 있는 문자들을 커버하기 위해 그들에 대한 "Control points of text polygon"을 정하고 최종적으로 "Polygin text instance"를 정함
- Architecture
  - ![architecture](https://miro.medium.com/max/1400/1*b6I-Bdj5itX7tllJ5HRKbg.png)

# PaddleOCR
- Reference: https://github.com/PaddlePaddle/PaddleOCR
## Install
```sh
# cpu
pip install paddlepaddle
# gpu
pip install paddlepaddle-gpu

pip install "paddleocr>=2.0.6" # 2.0.6 version is recommended
```