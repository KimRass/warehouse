from datetime import datetime
from pytz import timezone
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from tqdm import tqdm


def load_data(data_path):
    print(f"- Loading data '{data_path}'...", end=" ")
    data = pd.read_excel(data_path)
    print("Completed!")
    return data


def load_model(model_name):
    print(f"- Downloading pre-trained Sentence BERT model '{model_name}'...", end=" ")
    model = SentenceTransformer(model_name)
    print("Completed!")
    return model


def encode_texts_to_vectors(model, data, col):
    print("- Encoding each text to 768-dimensional vectors....")
    vecs = model.encode(data[col].tolist(), show_progress_bar=True, normalize_embeddings=True)
    print("Completed!")
    return vecs


def build_trees(vecs, n_trees):
    print(f"- Building a forest of {n_trees} trees for nearest neighbor search...", end=" ")
    dim = 768
    tree = AnnoyIndex(f=dim, metric="dot")
    for i, vec in enumerate(vecs):
        tree.add_item(i + 1, vec)
    tree.build(n_trees=n_trees, n_jobs=-1)
    print("Completed!")
    return tree


def get_similar_sentence_pairs(data, tree, n, threshold):
    print(f"- Searching up to {n} nearest neighbors for each sentence...")
    sim_sents = list()
    for id1 in tqdm(range(1, len(data) + 1)):
        ids, sims = tree.get_nns_by_item(i=id1, n=n, include_distances=True)
        for id2, sim in zip(ids, sims):
            if sim >= threshold and id1 < id2:
                sim_sents.append((id1, id2, sim))
    sim_sents = sorted(sim_sents, key=lambda x:x[2], reverse=True)
    df_sim_sents = pd.DataFrame(sim_sents, columns=["id1", "id2", "similarity"])

    id2text = {row["id"]:row["text"] for _, row in data.iterrows()}
    df_sim_sents.insert(2, "text1", df_sim_sents["id1"].map(id2text))
    df_sim_sents.insert(3, "text2", df_sim_sents["id2"].map(id2text))

    df_sim_sents.sort_values(by=["similarity", "id1", "id2"], ascending=[False, True, True], inplace=True)
    print("Completed!")
    return df_sim_sents


def save_data(df, save_to):
    print(f"- Saving the result as '{save_to}'...", end=" ")
    df.to_excel(save_to, index=False, encoding="euc-kr")
    print("Completed!")


def main():
    start = datetime.now(timezone("Asia/Seoul"))

    data = load_data("aihub_or_kr-sports_ko.xlsx")

    model = load_model("jhgan/ko-sroberta-sts")

    vecs = encode_texts_to_vectors(model, data, col="text")

    tree = build_trees(vecs, n_trees=8)

    threshold = 0.5
    df_sim_sents = get_similar_sentence_pairs(data, tree, n=5, threshold=threshold)

    save_data(df_sim_sents, save_to="Semantic_textual_similarity_Result.xlsx")

    end = datetime.now(timezone("Asia/Seoul"))

    print("- All precesses are done;")
    print(f"    - {'The program started at:':<24s}{datetime.strftime(start, format='%Y-%m-%d %H:%M:%S'):>20s}.")
    print(f"    - {'The program ended at:':<24s}{datetime.strftime(end, format='%Y-%m-%d %H:%M:%S'):>20s}.")

    elapsed = (end - start).total_seconds()
    print(f"    - {elapsed//60:,.0f}mins and {elapsed%60:,.0f}secs ({elapsed:,.0f}secs) elapsed.")
    print(f"    - {len(df_sim_sents):,} pair(s) of sentences showed a similarity of {threshold} or more.")


if __name__ == "__main__":
    main()
