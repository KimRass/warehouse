# `openpyxl`
```python
import openpyxl
```
## `openpyxl.Workbook()`
```python
wb = openpyxl.Workbook()
```
## `openpyxl.load_workbook()`
```python
wb = openpyxl.load_workbook("D:/디지털혁신팀/태블로/HR분석/FINAL/★직급별인원(5년)_본사현장(5년)-태블로적용.xlsx")
```
## `openpyxl.worksheet`
### `openpyxl.worksheet.formula`
#### `ArrayFormula`
```python
from openpyxl.worksheet.formula import ArrayFormula
```
```python
f = ArrayFormula("E2:E11", "=SUM(C2:C11*D2:D11)")
```
### `wb[]`
### `wb.active`
```python
ws = wb.active
```
### `wb.worksheets`
```python
ws = wb.worksheets[0]
```
### `wb.sheetnames`
### `wb.create_sheet()`
```python
wb.create_sheet("Index_sheet")
```
#### `ws[]`
#### `ws.values()`
```python
df = pd.DataFrame(ws.values)
```
#### `ws.insert_rows()`, `ws.insert_cols()`, `ws.delete_rows()`, `ws.delete_cols()`
#### `ws.append()`
```python
content = ["민수", "준공분", "거제2차", "15.06", "18.05", "1279"]
ws.append(content)
```
#### `ws.merge_cells()`, `ws.unmerge_cells()`
```python
ws.merge_cells("A2:D2")
```
### `wb.sheetnames`
### `wb.save()`
```python
wb.save("test.xlsx")
```
### `wb.active`
```python
sheet = wb.active
```
### `wb.save()`
```python
wb.save("test.xlsx")
```
#### `sheet.append()`
```python
content = ["민수", "준공분", "거제2차", "15.06", "18.05", "1279"]
sheet.append(content)
```
#### `sheet[]`
```python
sheet["H8"] = "=SUM(H6:H7)"
```

# `pptx`
## `Presentation()`
```python
from pptx import Presentation
```
```python
prs = Presentation("sample.pptx")
```
### `prs.slides[].shapes[].text_frame.paragraphs[].text`
```python
prs.slides[<<Index of the Slide>>].shapes[<<Index of the Text Box>>].text_frame.paragraphs[<<Index of the Text in the Text Box>>].text
```
### `prs.slides[].shapes[].text_frame.paragraphs[].font`
#### `prs.slides[].shapes[].text_frame.paragraphs[].text.name`
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].font.name = "Arial"
```
#### `prs.slides[].shapes[].text_frame.paragraphs[].text.size`
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].font.size = Pt(16)
```
### `prs.save()`
```python
prs.save("파일 이름")
```