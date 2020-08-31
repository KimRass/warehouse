# wordcloud
## WordCloud
```python
from wordcloud import WordCloud
```
```python
wc = WordCloud(font_path="C:/Windows/Fonts/HMKMRHD.TTF", relative_scaling=0.2, background_color="white", width=1600, height=1600, max_words=30000, mask=mask, max_font_size=80, background_color="white")
```
### wc.generate_from_frequencies()
```python
wc.generate_from_frequencies(words)
```
### wc.generate_from_text
### wc.recolor()
```python
wc.recolor(color_func=img_colors)
```
### wc.to_file()
```python
wc.to_file("test2.png")
```
## ImageColorGenerator
```python
from wordcloud import ImageColorGenerator
```
```python
img_arr = np.array(Image.open(pic))
img_colors = ImageColorGenerator(img_arr)
img_colors.default_color=[0.6, 0.6, 0.6]
```
## STOPWORDS
```python
from wordcloud import STOPWORDS
```
