# `kiwipiepy`
```python
# Reference: https://github.com/bab2min/kiwipiepy
from kiwipiepy import Kiwi

kiwi = Kiwi()

# `lm_search`: 둘 이상의 형태로 복원 가능한 모호한 형태소가 있는 경우, 이 값이 `True`면 언어 모델 탐색을 통해 최적의 형태소를 선택합니다. `False`일 경우 탐색을 실시하지 않지만 더 빠른 속도로 복원이 가능합니다.
kiwi.join(ls_token_new, [lm_search=True])

# `normalize_coda`: "ㅋㅋㅋ", "ㅎㅎㅎ"와 같은 초성체가 뒤따라와서 받침으로 들어갔을때 분석에 실패하는 문제를 해결해줍니다.
kiwi.tokenize(sentence, normalize_coda=True)
```

# POS List
```python
dic_kiwi_pos = {
	"N": ["NNG", "NNP", "NNB", "NR", "NP"],
	"V": ["VV", "VA", "VX", "VCP", "VCN"],
	"MA": ["MAG", "MAJ"],
	"J": ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"],
	"E": ["EP", "EF", "EC", "ETN", "ETM"],
	"XS": ["XSN", "XSV", "XSA"],
	"S": ["SF", "SP", "SS", "SE", "SO", "SW", "SL", "SH", "SN"],
	"W": ["W_URL", "W_EMAIL", "W_HASHTAG", "W_MENTION"]
}
```
