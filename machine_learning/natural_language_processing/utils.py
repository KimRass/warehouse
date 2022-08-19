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