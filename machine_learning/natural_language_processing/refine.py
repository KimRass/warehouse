import html

puncmark_map = {
	"“": "\"",
	"”": "\"",
	"ˮ": "\"",
	"″": "\"",
	"‘": "\'",
	"’": "\'",
	"ʼ": "\'",
	"´": "\'",
	"`": "\'",
	"ʻ": "\'",
	"，": ",",
}

nlp = spacy.load("en_core_web_sm")


def normalize_html_special_chars(text):
	return html.unescape(text)


def normalize_puncmarks(text):
	for old, new in puncmark_map.items():
		text = text.replace(old, new)
	return text