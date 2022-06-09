# Reference: https://huggingface.co/facebook/bart-large-mnli

import pandas as pd
from transformers import pipeline
import csv
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset

dataset = load_dataset("klue", "ynat", split="validation")

pd.options.display.max_colwidth = 200

tqdm.pandas()

# tags_csv_path = "tags.csv"
tags_csv_path = "/Users/jongbeom.kim/Desktop/workspace/Github/Data-Science/Machine-Learning/NLP/tags.csv"



# def get_tags(tags_csv_path):
#     df_tags = pd.read_csv(tags_csv_path)
#     return df_tags.columns.tolist()


def get_tags(classes_txt_path):
    with open(classes_txt_path, mode="r") as f:
        classes = f.readlines()
    classes = [i[:-1] for i in classes]
    return classes


def inference(model, sentence, classes):
    return model(sentence, classes, multi_label=True)


def get_model():
    model = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    return model


# def main(model, sentence):
    
#     infer_res = inference(model, sentence)
#     ls_tag = infer_res["labels"]
#     ls_conf = infer_res["scores"]

#     ser_tag_conf = pd.Series(ls_conf, index=ls_tag)

#     # ser_tag_conf.plot.barh()
#     # plt.show()
#     return ser_tag_conf

# df = pd.read_excel("/Users/jongbeom.kim/Downloads/valuesight_output/202203/news_eng_complete_202203.xlsx")[["sentence"]]

# for _, row in df.head(3).iterrows():
#     sentence = row["sentence"]
    # print(inference(model, sentence))
    
def func(x, classes):
    infer_res = inference(model, x, classes)
    label_most = infer_res["labels"][0]
    return label_most


df = pd.read_csv("/Users/jongbeom.kim/Desktop/workspace/CharCnn_Keras/data/ag_news_csv/test.csv", names=["label_index", "title", "sentence"]).sample(200, random_state=77)

classes_txt_path = "/Users/jongbeom.kim/Desktop/workspace/CharCnn_Keras/data/ag_news_csv/classes.txt"
classes = get_tags(classes_txt_path)

df["label_gt"] = df["label_index"].apply(lambda x: classes[x - 1])

df["label_pred"] = df["sentence"].progress_apply(lambda x: func(x, classes))
df["is_corr"] = df.apply(lambda x: True if x["label_gt"] == x["label_pred"] else False, axis=1)

print(f"Score: {df['is_corr'].sum()/len(df):.1%}")

dataset[0]