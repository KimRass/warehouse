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

# Reference: https://huggingface.co/facebook/bart-large-mnli

import pandas as pd
from transformers import pipeline
import csv
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset

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


def get_most_label(model, sentence, classes):
    infer_res = inference(model, sentence, classes)
    label_most = infer_res["labels"][0]
    return label_most


def evaluate_model_with_df(model, df, col_sentence, col_label, classes):
    df[col_label] = df[col_label].apply(lambda x: classes[x - 1])

    df["label_pred"] = df[col_sentence].progress_apply(lambda x: get_most_label(model, x, classes))
    df["is_corr"] = df.apply(lambda x: True if x[col_label] == x["label_pred"] else False, axis=1)

    score = df["is_corr"].sum() / len(df)
    return score


model = get_model()

df = pd.read_csv("/Users/jongbeom.kim/Desktop/workspace/CharCnn_Keras/data/ag_news_csv/test.csv", names=["label", "title", "sentence"]).sample(100, random_state=77)

classes_txt_path = "/Users/jongbeom.kim/Desktop/workspace/CharCnn_Keras/data/ag_news_csv/classes.txt"
classes = get_tags(classes_txt_path)
score = evaluate_model_with_df(model, df, "sentence", "label", classes)
print(f"Score: {score:.1%}")


ds = load_dataset("klue", "ynat", split="validation")
df_klue = ds.to_pandas()
df_klue = df_klue.sample(50, random_state=77)
df_klue = df_klue[["title", "label"]]
# df_klue.rename({"label": "label_index"}, axis=1, inplace=True)

classes = ds.info.features["label"].names
score = evaluate_model_with_df(model, df_klue, "title", "label", classes)
print(f"Score: {score:.1%}")