# Library Comparison
## PaddleOCR
- Source: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- 다양한 Text detection, text recongnition 모델을 지원합니다. ([Algorithms](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_overview_en.md))
- 자체 개발한 Text detection + Text recongnition 모델 'PP-OCRv3'를 제공합니다.
- 다수의 한자를 포함하여 3,687개의 문자를 지원하나 마침표와 쉼표 등이 없습니다. ([korean_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/korean_dict.txt))
## MMOCR
- Source: [MMOCR](https://github.com/open-mmlab/mmocr)
- 다양한 Text detection, text recongnition 모델을 지원합니다. ([Model Zoo](https://mmocr.readthedocs.io/en/latest/modelzoo.html))
## EasyOCR
- Source: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Text detection: [CRAFT](https://github.com/clovaai/CRAFT-pytorch) (Default), DBNet
- Text recognition
  - Transformation: None or TPS ([Thin Plate Spline](https://en.wikipedia.org/wiki/Thin_plate_spline))
  - Feature extraction: VGG, RCNN or ResNet
  - Sequence modeling: None or BiLSTM
  - Prediction: CTC or Attention
- Pre-trained model로서 사용할 [korean_g2](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.zip)의 구조는 'None-VGG-BiLSTM-CTC'입니다.
- 한국어에 대해서는 1,008개의 문자를 지원하고 한자는 없습니다.