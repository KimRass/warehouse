# Reference: https://huggingface.co/facebook/bart-large-mnli

import pandas as pd
from transformers import pipeline
import csv
import matplotlib.pyplot as plt

# tags_csv_path = "tags.csv"
tags_csv_path = "/Users/jongbeom.kim/Desktop/workspace/Github/Data-Science/Machine-Learning/NLP/tags.csv"


# def _get_tags(tags_csv_path):
#     f = open(tags_csv_path, 'r', encoding='utf-8')
#     rdr = csv.reader(f, delimiter=',')
#     return list(rdr)[0]

def get_tags(tags_csv_path):
    df_tags = pd.read_csv(tags_csv_path)
    return df_tags.columns.tolist()


def inference(model, sentence):
    tags = get_tags(tags_csv_path)
    return model(sentence, tags, multi_label=True)


def get_model():
    classifier = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli"
    )
    return classifier


def main(sentence):
    model = get_model()

    infer_res = inference(model, sentence)
    ls_tag = infer_res["labels"]
    ls_conf = infer_res["scores"]

    ser_tag_conf = pd.Series(ls_conf, index=ls_tag)

    ser_tag_conf.plot.barh()
    plt.show()


sentence = "우리가 어떤 업무를 수행할 때 단순히 우리의 기억과 논리에만 의존하는 것이 아니라, 여러 가지 도구를 활용합니다. 예를 들어, 숫자 계산의 정확도를 높이기 위해 계산기를 쓰고, 모르는 단어를 찾기 위해 사전을 찾고, 팩트 체크를 위해 검색 엔진이나 백과사전을 사용합니다. 적절한 시점에 도구를 활용해 생산성을 극대화하는 것이 좋은 결과물을 내기 위한 지름길이죠."
main(sentence)
