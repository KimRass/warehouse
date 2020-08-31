# pptx
## Presentation()
```python
from pptx import Presentation
```
```python
prs = Presentation("sample.pptx")
```
### prs.slides[].shapes[].text_frame.paragraphs[].text
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].text
```
- a : 슬라이드 번호
- b : 텍스트 상자의 인덱스
- c : 텍스트 상자 안에서 텍스트의 인덱스
### prs.slides[].shapes[].text_frame.paragraphs[].font
#### prs.slides[].shapes[].text_frame.paragraphs[].text.name
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].font.name = "Arial"
```
#### prs.slides[].shapes[].text_frame.paragraphs[].text.size
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].font.size = Pt(16)
```
### prs.save()
```python
prs.save("파일 이름")
```
