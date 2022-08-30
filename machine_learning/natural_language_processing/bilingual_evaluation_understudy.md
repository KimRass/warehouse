# BLEU (BiLingual Evaluation Understudy)
- References: https://en.wikipedia.org/wiki/BLEU, https://towardsdatascience.com/bleu-bilingual-evaluation-understudy-2b4eab9bcfd1
- ***BLEU is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" – this is the central idea behind BLEU.*** BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and remains one of the most popular automated and inexpensive metrics.
- ***Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations. Those scores are then averaged over the whole corpus to reach an estimate of the translation's overall quality. Intelligibility or grammatical correctness are not taken into account.***
- N-gram precision: (Number of n-grams from the cadidate found in any of the reference)/(The total number of n-grams in the candidate)
	```python
	def count_ngram(cand, n):
		return Counter(nltk.ngrams(cand, n))

	def ngram_precision(refs, cand, n):
		counter_refs = Counter()
		for ref in refs:
			counter_refs += count_ngram(ref, n)
		
		ngrams_cand = count_ngram(cand, n)
		tot_cnt = 0
		for ngram, cnt in ngrams_cand.items():
			if ngram in counter_refs:
				tot_cnt += cnt 
		return tot_cnt/len(cand) - n + 1
	```
- Modified n-grams precision
	- min(Number of n-grams from the candidate found in any of the reference, Maximum total count of n-grams in any of the reference)/(The total number of n-grams in the candidate)
	```python
	def max_ref_count(ngram, refs, n):
		maxim = 0
		for ref in refs:
			ngram2cnt_ref = count_ngram(ref, n)
			if ngram2cnt_ref[ngram] > maxim:
				maxim = ngram2cnt_ref[ngram]
		return maxim

	def count_clip(ngram, cand, refs, n):
		return min(count_ngram(cand, n)[ngram], max_ref_count(ngram, refs, n))

	def modified_ngram_precision(refs, cand, n):
		sum_count_clip = 0
		for ngram, cnt in count_ngram(cand, n).items():
			sum_count_clip += count_clip(ngram, cand, refs, n)
		return sum_count_clip/(len(cand) - n + 1)
	```
- *In practice, however, using individual words as the unit of comparison is not optimal. Instead, BLEU computes the same modified precision metric using n-grams. The length which has the "highest correlation with monolingual human judgements" was found to be four.*
- *To produce a score for the whole corpus, the modified precision scores for the segments are combined using the geometric mean multiplied by a brevity penalty to prevent very short candidates from receiving too high a score.* Let `r` be the total length of the reference corpus, and `c` the total length of the translation corpus. If `c<=r`, the brevity penalty applies, defined to be `np.exp(1 - r/c)`.
- *In the case of multiple reference sentences, `r` is taken to be the minimum of the lengths of the sentences whose lengths are closest to the lengths of the candidate sentences. ("best match length")*)
	```python
	def best_match_length(refs, cand):
		ref_lens = [len(ref) for ref in refs]
		return min(ref_lens, key=lambda x:(abs(x - len(cand)), x))

	def brevity_penalty(refs, cand):
		c = len(cand)
		r = best_match_length(refs, cand)

		if c == 0:
			return 0
		else:
			if c <= r:
				return np.exp(1 - r/c)
			else:
				return 1
	```
- Implementation
	```python
	def bleu_score(refs, cand, weights=[0.25, 0.25, 0.25, 0.25]):
		ps = [modified_ngram_precision(refs, cand, n=k + 1) for k, _ in enumerate(weights)]
		score = sum([w*np.log(p) if p != 0 else 0 for w, p in zip(weights, ps)])
		return brevity_penalty(refs, cand)*np.exp(score)
	```
- Using `nltk.translate.bleu_score.sentence_bleu()`
	- Reference: https://www.nltk.org/_modules/nltk/translate/bleu_score.html
	```python
	from nltk.translate.bleu_score import sentence_bleu
	
	# The default BLEU calculates a score for up to 4-grams using uniform
    weights (this is called BLEU-4).
	score = sentence_bleu(refs, cand, [weights=(0.25, 0.25, 0.25, 0.25)])
	```
- Using `nltk.translate.bleu_score.corpus_bleu()`
	- Reference: https://www.nltk.org/_modules/nltk/translate/bleu_score.html
	```python
	from nltk.translate.bleu_score import corpus_bleu()
	
	# The default BLEU calculates a score for up to 4-grams using uniform
    weights (this is called BLEU-4).
	score = corpus_bleu([refs], [cand], [weights=(0.25, 0.25, 0.25, 0.25)])
	```