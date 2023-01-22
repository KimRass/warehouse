# `MeCab`
- Install on Google Colab
	```python
	!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
	%cd Mecab-ko-for-Google-Colab
	!bash install_mecab-ko_on_colab190912.sh
	```
- On MacOS
	```python
	import MeCab

	mcb = MeCab.Tagger()

	mcb.parse
	```
	```python

- On Microsoft Windows
	```python
	# Reference: https://github.com/Pusnow/mecab-python-msvc/releases/tag/mecab_python-0.996_ko_0.9.2_msvc-2
	!pip install mecab_python-0.996_ko_0.9.2_msvc-cp37-cp37m-win_amd64.whl

	class Mecab:
		def pos(self, text):
			p = re.compile(".+\t[A-Z]+")
			return [tuple(p.match(line).group().split("\t")) for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
		
		def morphs(self, text):
			p = re.compile(".+\t[A-Z]+")
			return [p.match(line).group().split("\t")[0] for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
		
		def nouns(self, text):
			p = re.compile(".+\t[A-Z]+")
			temp = [tuple(p.match(line).group().split("\t")) for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
			nouns=[]
			for word in temp:
				if word[1] in ["NNG", "NNP", "NNB", "NNBC", "NP", "NR"]:
					nouns.append(word[0])
			return nouns
	mcb = Mecab()
	```