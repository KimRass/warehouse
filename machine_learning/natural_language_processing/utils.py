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


str_morphemes = convert_sentence_to_string_of_morphemes(sentence)
    while True:
        try:
            # idx_pattern = str_morphemes.index("EF") + 2
            pattern = r"EF\|[.,!?]ᴥSF\|"
            idx_pattern = re.search(pattern=pattern, string=str_morphemes).end()
            print(kiwi_join(str_morphemes[: idx_pattern]))
            str_morphemes = str_morphemes[idx_pattern:]
        except:
            print(kiwi_join(str_morphemes))
            break


def is_valid_parenthesis_string(sentence: str):
    stack = list()
    for char in sentence:
        if char == "(":
            stack.append(1)
        elif char == ")":
            if stack:
                stack.pop()
            else:
                return False
    else:
        return False if stack else True


def is_valid_quote_string(sentence: str):
    temp = sentence.count("'") % 2
    return True if temp == 0 else False


def is_valid_double_quote_string(sentence: str):
    temp = sentence.count('"') % 2
    return True if temp == 0 else False


def is_valid_string(sentence: str):
    is_valid = True if is_valid_parenthesis_string(sentence) and is_valid_quote_string(sentence) and is_valid_double_quote_string(sentence) else False
    return is_valid


def split_sentence_with_kiwi2(sentence):
    if sentence == "nan":
        return ""
    else:
        ls_sentence = kiwi.split_into_sents(
            sentence,
            normalize_coda=True,
            return_tokens=False
        )
        ls_sentence_new = list()
        sentence_merged = ""
        for i, sentence in enumerate(ls_sentence, start=1):
            sentence = sentence.text
            if is_valid_string(sentence) or i == len(ls_sentence):
                if sentence_merged != "":
                    ls_sentence_new.append(sentence_merged)
                    sentence_merged = ""
                ls_sentence_new.append(sentence)
            else:
                sentence_merged += sentence

        ls_sentence_new2 = list()
        for sentence in ls_sentence_new:
            sentence_without_puncmark = remove_puctuation_marks(sentence)
            if len(sentence_without_puncmark) <= 2:
                if ls_sentence_new2:
                    popped = ls_sentence_new2.pop()
                else:
                    popped = ""
                popped += sentence
                ls_sentence_new2.append(popped)
            else:
                ls_sentence_new2.append(sentence)
        return ls_sentence_new2


def get_list_of_single_quotes(sentence, searches="'"):
    ls_idx_tar_char = list()    
    for idx, char in enumerate(sentence):
        if char == searches:
            ls_idx_tar_char.append(idx)
    return cut_list_by_two(ls_idx_tar_char)


def check(sentence_start, sentence_end, idx1, idx2):
	is_in = (sentence_start <= idx1 and idx2 < sentence_end)
	is_out = (idx2 < sentence_start or sentence_end <= idx1)
	return is_in or is_out
    
    
def check_ls(sentence_start, sentence_end, ls_idx_tar_char):
    temp = {check(sentence_start, sentence_end, idx1, idx2) for idx1, idx2 in ls_idx_tar_char}
    return len(temp) == 1


def cut_list_by_two(ls):
    ls_new = list()
    for _ in range(len(ls) // 2):
        ls_new.append(ls[: 2])
        ls = ls[2:]
    return ls_new


def split_sentence_single_quote(sentence):
    ls_tup_sentence_start_sentence_end = [
        (sentence.start, sentence.end) for sentence in kiwi.split_into_sents(
            sentence, normalize_coda=True, return_tokens=False
        )
    ]
    ls_idx_tar_char = get_list_of_single_quotes(sentence)
    # print(ls_idx_tar_char)
    ls_sentence_new = list()
    for sentence_start, sentence_end in ls_tup_sentence_start_sentence_end:
        subsentence = sentence[sentence_start: sentence_end]
        condition = check_ls(sentence_start, sentence_end, ls_idx_tar_char)
        print(subsentence, condition)
        if not condition:
            if ls_sentence_new:
                popped = ls_sentence_new.pop()
            else:
                popped = ""
            popped += subsentence
            ls_sentence_new.append(popped)
        else:
            ls_sentence_new.append(subsentence)
    return ls_sentence_new


def split_sentence_single_quote(sentence, searches):
    ls_tup_sentence_start_sentence_end = [
        (sentence.start, sentence.end) for sentence in kiwi.split_into_sents(
            sentence, normalize_coda=True, return_tokens=False
        )
    ]
    ls_idx_tar_char = get_list_of_single_quotes(sentence, searches=searches)
    
    ls_tup_sentence_start_sentence_end_new = list()
    set_range_len = set(range(len(ls_tup_sentence_start_sentence_end)))
    for quote_start, quote_end in ls_idx_tar_char:
        ls_i = list()
        for i, (sentence_start, sentence_end) in enumerate(ls_tup_sentence_start_sentence_end):
            if sentence_start <= quote_start and quote_start < sentence_end:
                ls_i.append(i)
                set_range_len -= {i}
            if sentence_start <= quote_end and quote_end < sentence_end:
                ls_i.append(i)
                set_range_len -= {i}
        ls_tup_sentence_start_sentence_end_new.append(ls_i)        
    ls_tup_sentence_start_sentence_end_new = ls_tup_sentence_start_sentence_end_new + [[i]for i in set_range_len]
    ls_tup_sentence_start_sentence_end_new.sort(key=lambda x: x[0])
    
    ls_sentence = list()
    for sentence_start_sentence_end in ls_tup_sentence_start_sentence_end_new:
        temp = [ls_tup_sentence_start_sentence_end[j] for j in sentence_start_sentence_end]
        # print(temp)
        subsentence = sentence[temp[0][0]: temp[-1][-1]]
        ls_sentence.append(subsentence)
    return ls_sentence


def convert_sentence_to_string_of_morphemes(sentence):
    str_morphemes = "|".join(
        [f"{token.form}ᴥ{token.tag}" for token in kiwi.tokenize(sentence, normalize_coda=True)]
    )
    str_morphemes = "|" + str_morphemes + "|"
    return str_morphemes


def kiwi_join(str_morphemes):
    return kiwi.join(
        [tuple(str_morpheme.split("ᴥ")) for str_morpheme in str_morphemes.split("|") if str_morpheme != ""]
    )


def integrate_quotes_or_double_quotes(list_of_sentences, target_character="'"):
    temp = ""
    switch = False
    ls_tup_sentence_start_sentence_end_new = list()
    for s in list_of_sentences:
        if target_character in s:
            if switch:
                temp += " " + s
            switch = True if not switch else False
        if switch:
            temp += " " + s
        else:
            if temp:
                ls_tup_sentence_start_sentence_end_new.append(temp)
            else:
                ls_tup_sentence_start_sentence_end_new.append(s)
            temp = ""
    return ls_tup_sentence_start_sentence_end_new


def integrate_writtens(dir):
    dir = Path(dir)
    writtens_concated_path = dir / "writtens_concatenated.xlsx"
    if not writtens_concated_path.exists():
        writtens_dir = dir / "writtens"
        
        dfs = list()
        for writtens_path in sorted(writtens_dir.glob("*.xlsx")):
            df_writtens = pd.read_excel(writtens_path)
            dfs.append(df_writtens)
        df_sentences = pd.concat(dfs)
        df_sentences = add_converted_sentence_column(df_sentences)
        df_sentences = add_joined_sentence_column(df_sentences)

        df_sentences.to_excel(dir / "writtens_concatenated.xlsx", index=False)
        logger.info(f"Saved 'writtens_concatenated.xlsx'")
    else:
        logger.info(f"'{writtens_concated_path.name}' already exists.")


def remove_consecutive_identical_characters(transcript):
    ls_word = transcript.split()

    res = list()
    for word in ls_word:
        if word not in ["스스로", "많아지지"]:
            word_temp = ""
            for char in word:
                if not word_temp or word_temp[-1] != char:
                    word_temp += char
            word = word_temp
        res.append(word)
    return " ".join(res)


def treat_quotes(sentence):
    n_quote = sentence.count("'")
    n_double_quote = sentence.count('"')

    if (
        n_quote % 2 != 0 or
        n_double_quote % 2 != 0
    ):
        if n_quote != 0 and n_double_quote == 0:
            print(sentence)
            print("'" + sentence)
            sentence = "'" + sentence
        elif n_quote == 0 and n_double_quote != 0:
            sentence = '"' + sentence
        elif n_quote == 1 and n_double_quote == 1:
            sentence = sentence.replace('"', "'")
        else:
            sentence = "<수정 요망>"
    elif "''" in sentence:
        match = re.search(
            pattern=r"""(''[\w '".!?]+['"])(며|이라며|이라고|라며|라는|고)""", string=sentence
        )
        if match:
            sentence = sentence[:match.start()] + '"' + "'" + match.group(1)[2: -1] + '"' + match.group(2) + sentence[match.end():]
    return sentence