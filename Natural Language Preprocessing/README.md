# TF-IDF(Term Frequency-Inverse Document Frequency)
- Source: https://wikidocs.net/31698
- TF-IDF는 특정 문서에서 자주 등장하는 단어는 그 문서 내에서 중요한 단어로 판단

## tf(d, t)
- 특정 문서 d에서의 특정 단어 t의 등장 횟수.
- tf를 이어 붙인 것은 DTM과 같다.

## df(t)
- 특정 단어 t가 등장한 문서 d의 수.
- 여기서 특정 단어가 각 문서, 또는 문서들에서 몇 번 등장했는지는 관심가지지 않으며 오직 등장한 문서의 수에만 관심을 가집니다.

## idf(d, t)
- ![image.png](/wikis/2670857615939396646/files/2805180087290841767)
- n: 문서의 전체 개수.
