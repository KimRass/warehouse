import re

def split_sentence_based_on_initials(sentence):
    sentence_ori = sentence

    pattern = r"[ㄱ-ㅎㅏ-ㅣ]+"
    ls_idx = [0]
    while sentence:
        start, end = re.search(pattern, sentence).span()
        popped = ls_idx.pop()
        ls_idx.append(popped)
        ls_idx.append(popped + start)
        ls_idx.append(popped + end)
        sentence = sentence[end:]

    sentence_split = list()
    for i in range(len(ls_idx) - 1):
        idx_from = ls_idx[i]
        idx_to= ls_idx[i + 1]
        sentence_split.append(sentence_ori[idx_from: idx_to])
    # n_initials = sum(map(len, re.findall(pattern, sentence)))
    return sentence_split


def get_variations_for_word(word, original):
    ls_variation = list()
    ls_variation.append(re.sub(pattern=r"-|\^", repl="", string=word))

    original = str(original)
    if original != "nan":
        original = original.replace("←", "")
        original = original.lower()
        match = re.search(pattern=f"[a-z ]+", string=original)
        if match:
            original_en = match.__getitem__(0)
            if original_en == original:
                ls_variation.append(original)
    return ls_variation


def get_length_of_variation(variation):

    variation = "|가치| |중립|"
    len(variation.replace(" ", "").replace("|", ""))

    pattern = r"-|\^"
    len(re.sub(pattern=pattern, repl="", string="금융^사고"))


def categorize_words(dir) -> None:
    dir = Path(dir)

    words_categorized_dir = dir / "words_categorized"
    if len(list(words_categorized_dir.iterdir())) < 68:
        logger.info("Categorizing words...")
        
        words_categorized_dir.mkdir(exist_ok=True)
        
        words_concat_path = dir / "words_ko_concatenated.pkl"
        df_words_concated = pd.read_pickle(words_concat_path)

        for cat, group in tqdm(df_words_concated.groupby(["전문 분야"], dropna=False)):
            group.to_excel(dir / "words_categorized" / f"words_ko_{cat[1:-1]}.xlsx", index=False)

        logger.info("Completed categorizing words.")
    else:
        logger.info("words are already categorized.")


word = "세-계-시"
pattern = r"-|\^"
ls_substr = re.split(pattern=pattern, string=word)
ls_substr = list(map(lambda x: f"|{x}|", ls_substr))
n_replace = len(re.findall(pattern=pattern, string=word))
# original = "世界時"
# get_variations_for_word(word, original)
for variation in product(["", " "], repeat=n_replace):
    print(variation)